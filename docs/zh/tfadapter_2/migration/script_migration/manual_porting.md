# 手工迁移

本节主要介绍TensorFlow 2.6.5脚本迁移时涉及的迁移点及其作用或原因，您将在[手工迁移与训练](../sample_ref/manual_porting_and_training.md)章节看到具体的迁移示例以加深理解。如果迁移点中提及的部分API未在下文中出现，可在[TF Adapter 2.6接口参考](../../apiref/README.md)中查阅API详细说明。

通过本节内容，您可以将TensorFlow 2.6.5脚本迁移到AI处理器执行训练，功能正常后，如需进一步进行精度和性能调优，请参考[精度调试](../accuracy_debugging/accuracy_debugging.md)和[性能调优](../performance_tuning/performance_tuning.md)章节。

## 添加@tf.function装饰器

通常，官方发布或设计良好的脚本都会自带@tf.function装饰器或者允许您通过脚本入参控制，您无需额外添加。

特别地，使用Keras下Model.fit接口训练的脚本，TF2会在接口内部封装function，同样无需额外添加。

如果脚本中没有添加@tf.function装饰器，您可以先阅读TF2对@tf.function的使用说明，详情见[链接](https://www.tensorflow.org/api_docs/python/tf/function)，然后为您的训练/验证/推理函数添加@tf.function装饰器，并保证其在CPU或GPU环境下能正常工作。

## 设置NPU为默认设备

TF Adapter提供了注册NPU设备的API，将NPU注册为TensorFlow的合法设备。开发者可以在带有@tf.function装饰器并在CPU或GPU下可以正常工作的脚本文件开头，添加如下代码，设置NPU为默认设备。

```python
import npu_device as npu
# 默认设置Device 0为计算设备
npu.open().as_default()
```

您应当在import其他Python包前执行该操作，防止在加载后续包的过程中有未分发到NPU的算子执行行为。

npu.open接口的详细说明可参见[npu.open](../../apiref/npu-open.md)。

## 预处理batch动作设置drop_remainder

当原始网络脚本中使用dataset.batch\(batch_size\)返回动态形状时，由于数据流中剩余的样本数可能小于batch大小，导致网络中最后一个step的shape与之前的shape不一致，此种场景下会进入动态shape编译流程。为提升网络编译性能，建议将drop_remainder设置为True，丢弃文件中的最后几个样本，确保网络中每个step的shape一致。

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

关于batch操作的更多细节，请参考[链接](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)。

## 替换LossScaleOptimizer

### 使用前须知

- 针对Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，浮点计算的溢出模式默认采用“INF/NaN模式”，因此默认场景下，可直接跳过该迁移点。若您手工调用[set_device_sat_mode](../../apiref/set_device_sat_mode.md)接口将浮点计算的溢出模式修改为了“饱和模式”，则需要参考本章节进行脚本迁移。但需要注意，饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。
- 针对Atlas训练系列产品，如果您的原始脚本中未使用LossScaleOptimizer，可以直接跳过该迁移点；若您的原始脚本中使用了LossScaleOptimizer，请参考本章节进行脚本迁移。

### 迁移说明

通常，用户原始脚本在使用混合精度提升性能时，会使用LossScaleOptimizer保证精度，浮点计算溢出模式为“饱和模式”的场景下，由于NPU上浮点溢出的行为是全局标志置位而非产生Inf或NaN的输出，所以您需要使用NPU提供的[npu.train.optimizer.NpuLossScaleOptimizer](../../apiref/npu-train-optimizer-npulossscaleoptimizer.md)以获取正确的溢出检测结果。

npu.train.optimizer.NpuLossScaleOptimizer使用方法与tf.keras.mixed_precision.LossScaleOptimizer完全一致，使用细节可参考[链接](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer)。

如果您的脚本中使用的是tf.keras.mixed_precision.LossScaleOptimizer，直接替换为npu.train.optimizer.NpuLossScaleOptimizer即可。如果您使用了其他类型的LossScaleOptimizer，您应当先切换为tf.keras.mixed_precision.LossScaleOptimizer，进行功能精度验证后再进行上述替换。

## 训练循环下沉时设置NPU上的循环次数

该迁移点只要求设置一个**NPU上的循环下沉次数**，我们称之为**npu loop size**，有两种方法设置：

- 通过环境变量“NPU_LOOP_SIZE”设置，例如

    ```bash
    export NPU_LOOP_SIZE=32
    ```

    该变量需要在import npu_device前设置。

- 通过在您的训练脚本中调用[npu.set_npu_loop_size](../../apiref/npu-set_npu_loop_size.md)接口进行设置，所以需要用户理解**npu loop size**的含义。

**npu loop size**用于提升NPU训练的性能，在介绍**npu loop size**的由来前，先介绍TF2原生流程中的一些性能损耗点。以GPU为例，GPU环境常规训练方式下的工作时序，如下图所示，脚本侧用户控制执行十次训练，每次在GPU上执行一次训练步，训练步结束后，回到Python侧，用户判断当前步数等于10，启动下一次训练，直至训练完十次训练。

![](../figures/set_iterations_number_1.png)

观察时序图上的有色区域我们不难发现，此时不论是CPU还是GPU都是间歇性工作的，该模式下的问题：

- Python解释器存在额外开销，且运行耗时不稳定，两个训练步间的间隙造成性能黑洞。
- 数据预处理与GPU训练过程的流水不充分，虽然TF2 Dataset的prefetch功能可以消减预处理过程的耗时影响，但是每次训练时H2D（Host to Device）的数据传输以及CPU调度耗时是无法忽略的。

TF2为了省去Python解释器上的额外开销，推荐用户使用While算子来实现训练循环（也就是所谓的循环下沉，循环下沉并非NPU特有的策略），此时判断训练是否达到指定步数的逻辑不再在Python解释器中进行，而是依赖TF2中的While算子，在编码时，使用者应当这样组织自己的训练（下称“循环下沉的编码方式”）：

```python
@tf.function
def loop_train(iterator, steps):
    for i in tf.range(steps):
        train_step(next(iterator))
```

这样的TF2代码，在编译后，会将训练步嵌套在While算子中执行，时序变为下图所示：

![](../figures/set_iterations_number_2.png)

可以看出，采用循环下沉的策略后，在Python解释器上的耗时转移到TF CPU上，耗时更短也更稳定，但是在该形式下，仍然有两部分额外开销：

- 预处理H2D数据传输。
- 判定训练到达指定步数的算子计算耗时。

NPU为了达到较优性能，采取了两个策略来消除这两部分额外耗时：

- 异步预处理H2D线程，使得预处理输出传输与NPU训练完全异步，H2D的传输隐藏在NPU训练过程中。
- **需要用户指定训练循环下沉次数**，消除次数判断算子计算耗时（也用于指示预处理数据H2D异步传输次数）。

异步数据传输指TF Adapter的预处理线程主动向NPU发送训练数据，在未使用循环下沉的方式编码时，执行时序如下所示：

此时可以一定程度上消减数据预处理H2D数据传输与CPU调度的耗时（下发训练步的同时，数据传输正在进行）。

![](../figures/set_iterations_number_3.png)

当使用了训练循环下沉的编码方式时，NPU上的执行时序图为：

![](../figures/set_iterations_number_4.png)

可以看到：

- 脚本发起在NPU上训练十次的请求后，直到训练结束，都不会再与Python解释器交互，而是单纯的NPU运算。
- 预处理的耗时抖动，可以被前面训练步中预处理领先NPU运算的耗时抵消，从而可以抵御更大的数据预处理性能波动。

NPU训练循环下沉与异步预处理数据传输方式可以最大程度地减少训练计算无关的耗时，最大化性能收益，但同时对用户训练有额外约束：

由于预处理线程与NPU训练步异步，在使用循环下沉的编码方式时，需要告诉NPU当前的循环下沉次数，所以NPU要求用户在使用循环下沉的编码方式时，额外设置**npu loop size**，用于指示循环下沉执行的次数。

例如可以使用循环下沉的编码方式组织您的训练：

```python
@tf.function
def loop_train(iterator, steps):
    for i in tf.range(steps):
        train_step(next(iterator))
```

当您期望每次loop_train调用会在NPU上训练100个Steps时，此时您有两种方式设置**npu loop size**：

- 在启动训练前通过NPU_LOOP_SIZE的环境变量设置：

    ```bash
    export NPU_LOOP_SIZE=100
    ```

- 在Python脚本调用loop_train前调用[npu.set_npu_loop_size](../../apiref/npu-set_npu_loop_size.md)设置，然后就可以调用loop_train，并传入循环次数100：

    ```python
    npu.set_npu_loop_size(100)
    loop_train(train_iter, tf.constant(100))
    ```

您也可以在训练过程中调用npu.set_npu_loop_size来改变NPU上每次下沉执行的步数，假设总共训练100个Steps，希望每次在NPU上循环执行30Steps，此时最后的91\~100Steps小于**npu loop size**，所以可以在训练完90 Steps后调用[npu.set_npu_loop_size](../../apiref/npu-set_npu_loop_size.md)来调整**npu loop size**的大小：

```python
remaining_steps = 100  # 剩余步数
base_loop_size = 30  # 基准npu loop size
npu.set_npu_loop_size(base_loop_size)
while remaining_steps >= base_loop_size:  # 按照基准loop循环下沉训练，直到剩余Steps数不足一次loop
    loop_train(train_iterator, tf.constant(base_loop_size))
    remaining_steps -= base_loop_size
if remaining_steps > 0:  # 如果还有未处理的数据，调整为一个较小的npu loop size处理
    npu.set_npu_loop_size(remaining_steps)
    loop_train(train_iterator, tf.constant(remaining_steps))
```

## 分布式训练脚本适配（兼容单卡）

NPU上的分布式部署形态如下图所示，每个TensorFlow进程只管理独享的一张NPU训练卡，多个TensorFlow进程间，通过CANN提供的集合通信接口进行集群同步。单独观察某个worker，可以发现其与NPU上的单卡训练，除额外进行了集群内的集合通信外完全一致。

![](../figures/distributed_deploy_mode.png)

TF Adapter适配时，将单卡NPU视作集群worker数量为1的分布式部署形态，因而NPU的单卡脚本和分布式脚本最终是一致的。

NPU上执行分布式，相较于单卡NPU训练，主要有三部分的额外适配工作：

1. **worker间变量初值同步**

    TF2 Eager模式下，变量在模型生成后即完成初始化，此时需要进行变量初值同步操作，使各个worker上的变量初值一致。

    在模型构建完成后，您应当调用[npu.distribute.broadcast](../../apiref/npu-distribute-broadcast.md)接口完成变量初值同步，该接口要求传入需要进行worker间值同步的变量，通常，您可以通过model.trainable_variables来获取全部需要同步的变量。

2. **worker间梯度聚合**

    执行训练时，不同worker上产生不同的梯度信息grads，通过对多个worker上的梯度进行聚合计算，可以更准确地评估当前训练的误差情况。

    - 当原始脚本中分步骤计算并更新梯度（例如tf.gradients和opt.apply_gradients）时，则需要调用[npu.distribute.all_reduce](../../apiref/npu-distribute-all_reduce.md)接口完成梯度聚合运算，该接口要求您传入需要进行worker间聚合计算的梯度以及聚合运算的类型（通常是求平均值）。
    - 当原始脚本中计算和更新梯度操作被集成到同一接口中（例如minimize/model.fit）时，则需要调用[npu.distribute.npu_distributed_keras_optimizer_wrapper](../../apiref/npu-distribute-npu_distributed_keras_optimizer_wrapper.md)完成梯度聚合运算。

3. **不同worker上的数据集分片**

    分布式训练时，应当保证每个worker上评估的样本不同，这样才能使得训练结果更符合样本集真实分布，例如用户在一个8卡NPU的集群中执行训练，此时一个典型的策略就是第一张NPU卡上训练0-1/8的数据，第二张NPU卡训练1/8-2/8的数据，最后一张卡上训练7/8-8/8的数据。

    - 当数据集为tf.data.Dataset格式时，TF Adapter提供了[npu.distribute.shard_and_rebatch_dataset](../../apiref/npu-distribute-shard_and_rebatch_dataset.md)接口帮您实现上述切分动作，该接口要求您传入需要进行集群切分的Dataset（Dataset介绍参考[链接](https://www.tensorflow.org/guide/data)）以及集群训练时的全局batch大小，例如：

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

    - 当数据集为Numpy数组时，需要调用numpy方法手工对数据集和全局batch进行切分，例如：

        ```python
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data(os.path.join(args.data_path, 'mnist.npz'))

        # 根据设备数量均分数据集
        x_trains = np.split(x_train, args.rank_size)
        # 按设备编号取对应的数据集分片
        x_train = x_trains[args.device_id]
        x_tests = np.split(x_test, args.rank_size)
        x_test = x_tests[args.device_id]
        # 对全局batch进行切片
        batch_size = args.batch_size // args.rank_size

        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        ```

## 启动训练参数保持与单卡CPU形态一致

该迁移点通常要求您修改启动脚本时的分布式相关参数。

当前版本需要以单卡CPU的形态启动训练，即在AI处理器上启动训练时，保持与在单卡CPU上启动训练时的参数一致。

设计良好的脚本不会对部署形态做任何假设，配置脚本部署形态为单卡CPU通常只是启动脚本时入参的调整。

这里需要您评估脚本的启动参数，如果脚本支持传入分布式策略，请传入单卡分布式策略（OneDeviceStrategy）。如果支持配置GPU的数量，请配置为0。

采用此方案是因为：

- 单卡CPU的部署形态与NPU的单卡训练形态一致，TF Adapter能到分析较为纯净的训练过程，提升迁移成功率。
- 可以屏蔽原有脚本默认分布式策略的干扰，使得分布式迁移成功率大大提高。
