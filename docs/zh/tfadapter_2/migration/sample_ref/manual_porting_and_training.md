# 手工迁移与训练

## 下载TF2官方ResNet50

首先，我们下载TensorFlow的官方models仓并check out到v2.6.0的tag版本：

```bash
git clone https://github.com/tensorflow/models.git
cd models
git checkout v2.6.0
```

> [!NOTE]说明
> 在使用ResNet50之前，开发者需要对其脚本逻辑及执行参数有基本的了解。

## 添加@tf.function装饰器

我们分析入口文件official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py，可以看到官方脚本已经默认添加了@tf.function装饰器，因而这个迁移点我们直接迁移完成。

```python
flags.DEFINE_boolean(name='use_tf_function', default=True,
                     help='Wrap the train and test step inside a '
                     'tf.function.')
```

## 设置NPU为默认设备

在入口文件official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py的开头添加如下代码，添加完成后该迁移点即完成。

```python
import npu_device as npu
npu.open().as_default()
```

## 替换LossScaleOptimizer

脚本分析后，我们发现该脚本并未使用LossScaleOptimizer，因而不需要替换，这个迁移点直接完成。

## 预处理batch操作设置drop_remainder

跟随训练脚本逻辑，找到数据预处理文件：official/vision/image_classification/resnet/imagenet_preprocessing.py。

在数据读取函数“input_fn”内部设置drop_remainder为True，该迁移点完成。

```python
def input_fn(is_training,
             data_dir,
             batch_size,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False,
             training_dataset_cache=False,
             filenames=None):

"""
……
Returns:
  A dataset that can be used for iteration.
"""
drop_remainder=True

if filenames is None:
  filenames = get_filenames(is_training, data_dir)
dataset = tf.data.Dataset.from_tensor_slices(filenames)
```

## 设置NPU上的循环次数

在确定是否涉及该适配点前，需要您了解脚本是否采用了[循环下沉的编码方式](../script_migration/manual_porting.md#训练循环下沉时设置npu上的循环次数)，实际上，阅读官方的脚本发现已经开放了开关供用户选择是否循环下沉。

从`official/vision/image_classification/resnet/common.py`可以看到官方脚本提供了两个入参：

- steps_per_loop传入training loop的大小，可以从注释中看出，在循环中间，只有训练的动作，不会执行任何callback之类的附加操作。
- use_tf_while_loop则决定是否循环下沉，默认为True，即training loop默认都会以While算子的形式执行。

```python
flags.DEFINE_integer(
    name='steps_per_loop',
    default=None,
    help='Number of steps per training loop. Only training step happens '
    'inside the loop. Callbacks will not be called inside. Will be capped at '
    'steps per epoch.')
flags.DEFINE_boolean(
    name='use_tf_while_loop',
    default=True,
    help='Whether to build a tf.while_loop inside the training loop on the '
    'host. Setting it to True is critical to have peak performance on '
    'TPU.')
```

所以，按照默认值，我们需要设置NPU_LOOP_SIZE环境变量的值与steps_per_loop一致。此环境变量的设置说明可参见[启动单卡训练](#启动单卡训练)。

## 启动单卡训练

我们暂时略过了[分布式训练脚本适配（兼容单卡）](../script_migration/manual_porting.md#分布式训练脚本适配兼容单卡)的分布式适配，初步验证下单卡迁移的结果。

首先，我们需要确定启动脚本的参数，为了以CPU单卡形式启动脚本，我们传入distribution_strategy的策略为one_device。

在启动前，我们需要按照official/vision/image_classification/resnet/README.md中的说明，将models路径设置到PYTHONPATH中，例如当前所在目录是/path/to/models，则应当设置环境变量：

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

训练中，我们通常每次循环下沉执行一个epoch数据的训练，steps_per_loop值则应当设置为样本总数除以batch大小的结果，为了快速验证功能，我们假定样本总数为64，训练的batch大小为2，同时跳过eval过程。所以此时steps_per_loop的大小为64/2=32，因此我们需要设置环境变量export NPU_LOOP_SIZE=32。最终我们的启动参数如下（其中**/path/to/imagenet_TF/需要替换为您的数据集路径**），需要注意的是，通常您应当以epoch为单位组织训练，这里入参中写入train_steps是为了使训练尽快结束进行基本的功能验证。

```bash
cd official/vision/image_classification/resnet/
export PYTHONPATH=$PYTHONPATH:/path/to/models
export NPU_LOOP_SIZE=32
python3 resnet_ctl_imagenet_main.py \
--data_dir=/path/to/imagenet_TF/ \
--train_steps=128 \
--distribution_strategy=one_device \
--use_tf_while_loop=true \
--steps_per_loop=32 \
--batch_size=2 \
--epochs_between_evals=1 \
--skip_eval
```

## 关键日志说明

NPU上的单卡训练将很快完成，下面对执行日志的关键位置进行说明。

### 关键日志1：NPU初始化配置项及初始化成功打印

这部分日志可以看到NPU初始化时的配置项（如果您修改了[npu.global_options](../../apiref/npu-global_options/README.md)，这里也会有所体现）以及初始化成功信息，由于我们调用[npu.open](../../apiref/npu-open.md)时没有传入任何参数，所以默认在NPU:0上进行了初始化。

![](../figures/npu_init_success.png)

### 关键日志2：数据预处理H2D线程开启及HDC通道创建

这部分日志只有在您使用了Dataset作为预处理Pipeline，并且function的入参是Iterator时才会打印。

从这两条日志上可以看出，TF Adapter启动了预处理H2D线程，同时创建了名为AnonymousIterator0的HDC数据传输通道（该名称与TF2中Iterator的shared_name一致）。

![](../figures/data_preprocess_result1.png)

![](../figures/data_preprocess_result2.png)

### 关键日志3：NPU检测到循环下沉逻辑，以循环下沉方式执行训练

这个日志的打印内容取决于您是否采用了[循环下沉的编码方式](../script_migration/manual_porting.md#训练循环下沉时设置npu上的循环次数)，本次迁移采用了该编码方式，所以Graph xxx can loop的判定结果是true，同时设置了NPU上的训练循环大小为32。后两行日志可以看出，开启了总共32次的异步数据传输，然后向NPU设备下发了执行32次训练的请求。

![](../figures/train_exec_result1.png)

### 关键日志4：训练执行过程

这部分日志表示训练过程正在执行。

![](../figures/train_exec_result2.png)

其中，下面的日志表示32次异步的数据传输成功完成，如果数据传输过程中出现错误，也可以看到相关打印。

![](../figures/train_exec_result3.png)

### 关键日志5：训练进程退出

训练结束后，进程退出，会依次销毁HDC数据传输线程（如果存在），关闭Graph Engine引擎。

![](../figures/train_over_result.png)

## 分布式适配

在单卡模式下已经成功完成迁移，我们开始分布式模式的迁移，分布式迁移并不影响单卡功能，您的分布式脚本和单卡脚本最终是同一份脚本，可以同时以单卡或分布式方式执行。我们依次按照[分布式迁移](../script_migration/manual_porting.md#分布式训练脚本适配兼容单卡)的过程开始迁移。

1. worker间变量初值同步。

    根据脚本逻辑找到模型创建完成的位置official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py，添加可训练变量的同步操作，这里需要使用[npu.distribute.broadcast](../../apiref/npu-distribute-broadcast.md)接口。

    ```python
    with distribute_utils.get_strategy_scope(strategy):
      # 模型创建
      runnable = resnet_runnable.ResnetRunnable(flags_obj, time_callback, per_epoch_steps)
    # 变量同步
    npu.distribute.broadcast(runnable.model.trainable_variables)
    ```

2. worker间梯度聚合。

    根据脚本逻辑，我们找到训练过程中梯度更新的部分official/vision/image_classification/resnet/resnet_runnable.py

    ```python
    def train_step(self, iterator):
      """See base class."""

      def step_fn(inputs):
        """Function to run on the device."""
        images, labels = inputs
        with tf.GradientTape() as tape:
          logits = self.model(images, training=True)

          prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, logits)
          loss = tf.reduce_sum(prediction_loss) * (1.0 /
                                                   self.flags_obj.batch_size)
          num_replicas = self.strategy.num_replicas_in_sync
          l2_weight_decay = 1e-4
          if self.flags_obj.single_l2_loss_op:
            l2_loss = l2_weight_decay * 2 * tf.add_n([
                tf.nn.l2_loss(v)
                for v in self.model.trainable_variables
                if 'bn' not in v.name
            ])

            loss += (l2_loss / num_replicas)
          else:
            loss += (tf.reduce_sum(self.model.losses) / num_replicas)

        grad_utils.minimize_using_explicit_allreduce(
            tape, self.optimizer, loss, self.model.trainable_variables)
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, logits)
    ```

    可以看出，TF2原始脚本中，使用了函数minimize_using_explicit_allreduce来屏蔽部署形态，进入函数内部，可以找到实际执行梯度聚合的函数在：official/staging/training/grad_utils.py。

    ```python
    def _filter_and_allreduce_gradients(grads_and_vars,
                                        allreduce_precision="float32",
                                        bytes_per_pack=0):
    ```

    需要注意，我们要求以[单卡CPU的形式](../script_migration/manual_porting.md#启动训练参数保持与单卡cpu形态一致)启动训练，所以，此时代码中的原始梯度聚合行为不生效（或者说因为是单卡，所以不需要聚合），在这个函数内部，我们需要添加NPU上的梯度聚合操作，这里需要用到[npu.distribute.all_reduce](../../apiref/npu-distribute-all_reduce.md)接口。我们在official/staging/training/grad_utils.py添加如下信息：

    ```python
    # 由于需要使用npu.distribute.all_reduce接口，在脚本开头import npu
    import npu_device as npu

    def _filter_and_allreduce_gradients(grads_and_vars,
                                        allreduce_precision="float32",
                                        bytes_per_pack=0):
    ... ...

      # 原始脚本采用SUM策略
      allreduced_grads = tf.distribute.get_strategy(  # pylint: disable=protected-access
      ).extended._replica_ctx_all_reduce(tf.distribute.ReduceOp.SUM, grads, hints)
      if allreduce_precision == "float16":
        allreduced_grads = [tf.cast(grad, "float32") for grad in allreduced_grads]

      # 由于NPU适配添加的梯度聚合操作，聚合类型保持与原始脚本一致，此处选择“sum”聚合策略
      allreduced_grads = npu.distribute.all_reduce(allreduced_grads,reduction="sum")

      return allreduced_grads, variables
    ```

3. 不同worker上的数据集分片。

    根据脚本逻辑找到预处理函数official/vision/image_classification/resnet/resnet_runnable.py：

    ```python
        # 假数据，忽略该分支
        if self.flags_obj.use_synthetic_data:
          self.input_fn = common.get_synth_input_fn(
              height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
              width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
              num_channels=imagenet_preprocessing.NUM_CHANNELS,
              num_classes=imagenet_preprocessing.NUM_CLASSES,
              dtype=self.dtype,
              drop_remainder=True)
        else:
        # 真实的预处理方法
          self.input_fn = imagenet_preprocessing.input_fn
    ```

    找到official/vision/image_classification/resnet/imagenet_preprocessing.py，添加如下信息：

    ```python
    # 由于需要使用npu.distribute.shard_and_rebatch接口，在脚本开头import npu
    import npu_device as npu

    if input_context:
      logging.info(
          'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
          input_context.input_pipeline_id, input_context.num_input_pipelines)
      # 原始的shard逻辑，因为以单机CPU方式启动，所以不会进行实际的shard
      dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    # NPU添加的shard逻辑，会根据集群数量，对数据集和全局batch进行切分
    dataset, batch_size = npu.distribute.shard_and_rebatch_dataset(dataset, batch_size)
    ```

执行完上述步骤后，分布式迁移完成。

> [!NOTE]说明
> 新增的调用对单卡流程没有任何影响，这些接口内部会根据是否设置了NPU分布式执行的环境变量来决定是否生效。如果是单卡模式执行，这些接口不会执行任何操作。

## 启动分布式训练

我们已经完成了分布式的迁移工作，不过此时您仍然可以使用上面的单卡参数进行单卡训练。

如果要进行分布式训练，还需要调整一些额外的启动参数，需要注意的是，这些调整并非NPU特有的：当使用多卡训练时，可以对全局batch size进行等比例的放大，例如在单卡上执行32 batch size大小的训练，在执行集群大小为8的分布式训练时，batch size大小可以调整为32\*8以加速训练。

我们以上面启动单卡的参数为例，单卡batch大小为2，所以8卡训练时将batch大小调整为2\*8=16，batch的调整会直接影响每个epoch的训练总步数，您应当清楚这些关联关系，并且在batch大小发生变化时作出调整。假定一个epoch样本总数为64，我们每次循环下沉处理一个epoch，那么，当batch大小为2时，steps_per_loop设置为64/2=32，表示单卡上训练32步即完成了一个epoch训练。但是当我们使用8卡训练，batch大小调整为16后，steps_per_loop应当设置为64/16=4，此时，单卡上训练4步即完成了一个epoch训练，单纯从步数上可以看出有8倍的性能提升。

清楚这些关系后，我们将单卡的参数进行转换，变为分布式的启动参数，由于要启动多个训练进程，可以将启动命令行写入脚本，下面的8卡训练脚本仅供参考，例如命名为train.sh。

```bash
export RANK_TABLE_FILE=/path/to/rank_table.json
export RANK_SIZE=8
export RANK_ID=$1
export ASCEND_DEVICE_ID=$2
export NPU_LOOP_SIZE=4
python3 resnet_ctl_imagenet_main.py \
--data_dir=/path/to/imagenet_TF/ \
--train_steps=16 \
--distribution_strategy=one_device \
--use_tf_while_loop=true \
--steps_per_loop=4 \
--batch_size=16 \
--epochs_between_evals=1 \
--skip_eval
```

> [!NOTE]说明
>
> - /path/to/rank_table.json替换为符合您部署形态的NPU分布式配置文件。
> - /path/to/imagenet_TF/替换为您实际的数据集路径。
> - 此样例，我们以通过配置文件（即rank table文件）的方式配置AI处理器的资源信息，详细的配置文件说明可参见《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“相关参考 \> 集群信息配置”章节。当然，开发者也可以通过环境变量的方式指定AI处理器的资源信息，可参见[训练执行（通过环境变量配置资源信息）](../model_training/distributed_training.md#训练执行通过环境变量配置资源信息)。

在启动前，我们同样需要按照official/vision/image_classification/resnet/README.md中的说明，将models路径设置到PYTHONPATH中，例如当前的目录是/path/to/models，环境变量示例如下：

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

之后，您可以执行如下命令启动8卡NPU训练：

```text
nohup bash train.sh 0 0 &
nohup bash train.sh 1 1 &
nohup bash train.sh 2 2 &
nohup bash train.sh 3 3 &
nohup bash train.sh 4 4 &
nohup bash train.sh 5 5 &
nohup bash train.sh 6 6 &
nohup bash train.sh 7 7 &
```
