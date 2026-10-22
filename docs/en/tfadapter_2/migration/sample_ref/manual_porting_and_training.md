# Manual Porting and Training

## Downloading TF2 ResNet-50

Clone TensorFlow's models repository and check out to the release tagged v2.6.0.

```bash
git clone https://github.com/tensorflow/models.git
cd models
git checkout v2.6.0
```

> [!NOTE]NOTE
> Before using ResNet-50, you need to have a basic understanding of the script logic and execution parameters.

## Adding the @tf.function Decorator

As the entry point file  **official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py**  is already added with the  **@tf.function**  decorator, this step can be safely skipped.

```python
flags.DEFINE_boolean(name='use_tf_function', default=True,
                     help='Wrap the train and test step inside a '
                     'tf.function.')
```

## Setting NPU as Default Device

Add the following code lines at the beginning of the entry point file  **official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py**.

```python
import npu_device as npu
npu.open().as_default()
```

## Replacing LossScaleOptimizer

As the script does not involve the use of LossScaleOptimizer, this step can be safely skipped.

## Setting drop_remainder for Static Batching

Navigate to the data preprocessing file  **official/vision/image_classification/resnet/imagenet_preprocessing.py**  based on the training script logic.

Set  **drop_remainder**  to  **True**  in the data read function  **input_fn**  to complete the porting.

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
......
Returns:
  A dataset that can be used for iteration.
"""
drop_remainder=True

if filenames is None:
  filenames = get_filenames(is_training, data_dir)
dataset = tf.data.Dataset.from_tensor_slices(filenames)
```

## Setting the Number of Iterations Offloaded to NPU

Determine if  [iteration offload](../script_migration/manual_porting.md#setting-the-number-of-iterations-offloaded-to-npu)  is implemented in the script.

The  **official/vision/image_classification/resnet/common.py**  script provides two input parameters:

- **steps_per_loop**  indicates the training loop size. As the comment suggests, only training steps are performed inside a loop, with no additional operations such as callbacks.
- **use_tf_while_loop**  defaults to  **True**, meaning that iteration offload is enabled and each training loop is executed as operator While.

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

Set the  **NPU_LOOP_SIZE**  environment variable according to the  **steps_per_loop**  argument. For details about how to set the environment variable, see  [Starting Single-Device Training](#starting-single-device-training).

## Starting Single-Device Training

We skip  [Distributed Training Script Adaptation \(Single Device\)](../script_migration/manual_porting.md#using-training-startup-parameters-consistent-with-single-cpu-training)  to preliminarily validate the single-device porting result.

First, determine the parameters in the startup script. To use the same parameters for single CPU training, set  **distribution_strategy**  to  **one_device**.

Add the  **models**  directory to  **PYTHONPATH**  according to the description in  **official/vision/image_classification/resnet/README.md**. The following example assumes that the current directory is  **/path/to/models**:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

As a best practice, we offload a training epoch to the device in iteration offload mode. Set  **steps_per_loop**  to the dataset size divided by the batch size. If the dataset size is 64 and batch size is 2 and the evaluation phase is not wanted,  **steps_per_loop**  should be  **32**  \(= 64/2\) and therefore the environment variable should be set using  **export NPU_LOOP_SIZE=32**. The final startup parameters are as follows \(replace  **/path/to/imagenet_TF/**  with your dataset directory\). Training is organized by epoch in normal cases. The argument  **train_steps**  is used here only for validation convenience.

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

## What and Where Information Is Logged

This section describes what and where information is logged during single NPU training.

### 1. NPU Initialization Configurations and Initialization Success

The log shows the NPU initialization configurations, including any update to  [npu.global_options](../../apiref/npu-global_options/README.md), and the initialization success message. As no argument is passed to the  [npu.open](../../apiref/npu-open.md)  call, initialization is performed on  **NPU:0**  by default.

![](../figures/npu_init_success.png)

### 2. Preprocessing H2D Thread Started and HDC Channel Created

The log is printed only when you use  **Dataset**  as the preprocessing pipeline and  **Iterator**  is passed to the function call.

In this log example, the TF Adapter starts the preprocessing H2D thread and creates an HDC channel named  **AnonymousIterator0**. The channel name is the same as the value of  **shared_name**  of the  **Iterator**  in TF2.

![](../figures/data_preprocess_result1.png)

![](../figures/data_preprocess_result2.png)

### 3. Iteration Offload to NPU Detected and Training Started

If  [iteration offload](../script_migration/manual_porting.md#setting-the-number-of-iterations-offloaded-to-npu) is enabled,  **Graph xxx can loop**  will be determined as  **true**. In this example, the number of training iterations \(loop size\) to be offloaded to the NPU is set to  **32**. As the log suggests, 32 asynchronous data transfers are started and 32 training requests are issued.

![](../figures/train_exec_result1.png)

### 4. Training Progress

The log indicates the training process.

![](../figures/train_exec_result2.png)

The following message indicates that 32 asynchronous data transfers are completed successfully. If any transfer error occurs, an error message will be printed.

![](../figures/train_exec_result3.png)

### 5. Training Process Exited

After the training is complete, the process will exit, the created HDC data transfer thread will be destroyed, and Graph Engine will be switched off.

![](../figures/train_over_result.png)

## Adapting to Distributed Setup

This step allows you to run single-device training and distributed training using the same script. Note that distributed adaptation has zero impact on the single-device training process. The  [porting workflow for distributed training](../script_migration/manual_porting.md#distributed-training-script-adaptation-single-device)  goes through the following steps:

1. Synchronizing initial values of variables between workers.

    In  **official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py**, add the action of synchronizing trainable variables by inserting the  [npu.distribute.broadcast](../../apiref/npu-distribute-broadcast.md)  API.

    ```python
    with distribute_utils.get_strategy_scope(strategy):
      # Model creation
      runnable = resnet_runnable.ResnetRunnable(flags_obj, time_callback, per_epoch_steps)
    # Variable synchronization
    npu.distribute.broadcast(runnable.model.trainable_variables)
    ```

2. Aggregating gradients between workers.

    Find the  **official/vision/image_classification/resnet/resnet_runnable.py**  script.

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

    In the source TF2 script, the  **minimize_using_explicit_allreduce**  function is used to shield the setup form and the function for executing gradient aggregation is implemented in  **official/staging/training/grad_utils.py**.

    ```python
    def _filter_and_allreduce_gradients(grads_and_vars,
                                        allreduce_precision="float32",
                                        bytes_per_pack=0):
    ```

    The TF Adapter requires that training is started in  [single-CPU training form](../script_migration/manual_porting.md#using-training-startup-parameters-consistent-with-single-cpu-training). As single-device training does not involve aggregation, the original gradient aggregation code will not be executed. In this function, add the NPU gradient aggregation action by calling the  [npu.distribute.all_reduce](../../apiref/npu-distribute-all_reduce.md)  API. Add the following lines to the  **official/staging/training/grad_utils.py**  file:

    ```python
    # Import npu at the beginning of the script to use the npu.distribute.all_reduce API.
    import npu_device as npu

    def _filter_and_allreduce_gradients(grads_and_vars,
                                        allreduce_precision="float32",
                                        bytes_per_pack=0):
    ... ...

    # The original script uses the SUM strategy.
      allreduced_grads = tf.distribute.get_strategy(  # pylint: disable=protected-access
      ).extended._replica_ctx_all_reduce(tf.distribute.ReduceOp.SUM, grads, hints)
      if allreduce_precision == "float16":
        allreduced_grads = [tf.cast(grad, "float32") for grad in allreduced_grads]

    # Gradient aggregation added due to NPU adaptation. Keep it consistent with that of the original script, that is, "sum".
      allreduced_grads = npu.distribute.all_reduce(allreduced_grads,reduction="sum")

      return allreduced_grads, variables
    ```

3. Sharding dataset to workers.

    Find the preprocessing function in  **official/vision/image_classification/resnet/resnet_runnable.py**.

    ```python
        # Fake data. Ignore this branch.
        if self.flags_obj.use_synthetic_data:
          self.input_fn = common.get_synth_input_fn(
              height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
              width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
              num_channels=imagenet_preprocessing.NUM_CHANNELS,
              num_classes=imagenet_preprocessing.NUM_CLASSES,
              dtype=self.dtype,
              drop_remainder=True)
        else:
        # Actual preprocessing method
          self.input_fn = imagenet_preprocessing.input_fn
    ```

    Add the following code to  **official/vision/image_classification/resnet/imagenet_preprocessing.py**:

    ```python
    # Import npu at the beginning of the script to use the npu.distribute.shard_and_rebatch API.
    import npu_device as npu

    if input_context:
      logging.info(
          'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
          input_context.input_pipeline_id, input_context.num_input_pipelines)
      # Original shard logic. Shard is not performed, as training is performed in single-CPU mode.
      dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    # Shard logic added by the NPU. The dataset and global batch will be sharded based on the number of clusters.
    dataset, batch_size = npu.distribute.shard_and_rebatch_dataset(dataset, batch_size)
    ```

After the preceding steps are performed, the porting for distributed training is complete.

> [!NOTE]NOTE
> These APIs take effect depends on whether the environment variables for NPU distributed training are set. For single-device training, these APIs will not take effect.

## Starting Distributed Training

You can run single-device training and distributed training using the same script.

Some extra tweaking on the startup parameters is needed to run distributed training. However, such tweaking is not unique to NPU training. You always need to increase the global batch size proportionally when you scale up the training devices. For example, a batch size of 32 for single-device training can be scaled to 32 × 8 for an 8-device cluster to accelerate training.

The preceding single-device training example uses batch size 2. We increase the batch size to 16 \(= 8 × 2\) in eight-device training. The change of batch size directly affects the total number of training steps per epoch. Assume that 64 samples are passed every epoch. When the batch size is 2, set  **steps_per_loop**  to  **32**  \(= 64/2\), indicating that every 32 steps on a single device complete one training epoch. However, in 8-device training, the batch size is increased to  **16**  and therefore  **steps_per_loop**  should be set to  **4**  \(= 64/16\), indicating that every 4 steps on a single device complete one training epoch — an 8x performance boost.

In distributed training, as multiple training processes will be started, you can write the startup command line into the script. The following 8-device training script \(for example,  **train.sh**\) is for reference only.

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

> [!NOTE]NOTE
>
> - Replace  **/path/to/rank_table.json**  with the NPU distributed configuration file that meets your setup requirements.
> - Replace  **/path/to/imagenet_TF/**  with the actual dataset directory.
> - In this example, the resource information of the  AI processor  is configured in the configuration file \(that is, a rank table file\). For details about the configuration file, see  Reference \> Cluster Information Configuration  in  [Huawei Collective Communication Library \(HCCL\)](https://hiascend.com/en/document/redirect/CannCommunityHcclUg). Alternatively, you can use environment variables to specify resource information of the  AI processor. For details, see  [Training Execution \(Setting Environment Variables\)](../model_training/distributed_training.md#training-execution-configuring-resources-via-environment-variables).

Add the  **models**  directory to  **PYTHONPATH**  according to the description in  **official/vision/image_classification/resnet/README.md**. For example, if the  **models**  directory is  **/path/to/models**, set the environment variable as follows:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

Next, run the following commands to start 8-device NPU training.

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
