# Manual Porting

This section highlights the key points in porting TensorFlow 2.6.5 scripts. If you want to get quick experience of the whole process, see  [Manual Porting and Training](../sample_ref/manual_porting_and_training.md). If some APIs mentioned in the porting are not described in the following sections, see  [TF Adapter 2.6 API Reference](../../apiref/README.md)  for more information.

By learning this section, you will learn how to port TensorFlow 2.6.5 scripts to the AI processor for training and functionality check. For details about accuracy and performance tuning, see [Accuracy Tuning](../accuracy_debugging/accuracy_debugging.md) and [Performance Tuning](../performance_tuning/performance_tuning.md).

## Adding the @tf.function Decorator

Generally, you do not have to manually add the  **@tf.function**  decorator if you are using an official or well-designed script.

Particularly, if your script uses Keras's  **Model.fit**  API for model training, you can safely skip this step as the function is already encapsulated in the API.

If your script does not have the  **@tf.function**  decorator, refer to  [TF2 guide on @tf.function](https://www.tensorflow.org/api_docs/python/tf/function)  first before you add  **@tf.function**  to your training/validation/inference function and validate the function in the CPU or GPU environment.

## Setting NPU as Default Device

The TF Adapter provides an API for registering an NPU as a valid TensorFlow device. You can add the following code to the beginning of the script file that has the  **@tf.function**  decorator and can work properly under the CPU or GPU to set an NPU as the default device:

```python
import npu_device as npu
# By default, device 0 is a compute device.
npu.open().as_default()
```

Perform this operation before importing other Python packages. Otherwise, operators may be executed on devices other than the NPU during the loading of subsequent packages.

For details about  **npu.open**, see  [npu.open](../../apiref/npu-open.md).

## Setting drop_remainder for Static Batching

If the original network script relies on  **dataset.batch\(batch_size\)**  to return the dynamic shape, the shape of the last step on the network may be inconsistent with the previous shape because the number of remaining samples in the data flow may be less than the batch size. In this scenario, the dynamic shape compilation process starts. To improve network compilation performance, you are advised to set  **drop_remainder**  to  **True**  to discard the last several samples in the file and ensure that the shape of each step on the network is the same.

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

Click  [here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)  for more information.

## Replacing LossScaleOptimizer

### Precautions

- For the  Ascend 950PR&950DT products, Atlas A3 products, Atlas A2 products, the overflow/underflow mode of floating-point computation uses the INF/NaN mode by default. Therefore, you can skip this step. If you have manually called the  [set_device_sat_mode](../../apiref/set_device_sat_mode.md)  API to change the overflow/underflow mode to the saturation mode, you need to port scripts by referring to this section. Note that the saturation mode is only compatible with earlier versions and will not be evolved in the future. In addition, the compute in this mode may be inaccurate.
- For the  Atlas training products, skip this step if your script does not involve the use of LossScaleOptimizer. Otherwise, port the script by referring to this section.

### Description

Generally, LossScaleOptimizer is used to preserve accuracy while improving performance with mixed precision. In saturation mode, NPU floating-point overflow sets a global flag instead of producing Inf or NaN, so use the NPU-provided [npu.train.optimizer.NpuLossScaleOptimizer](../../apiref/npu-train-optimizer-npulossscaleoptimizer.md) to obtain correct overflow detection results.

The usage of  **npu.train.optimizer.NpuLossScaleOptimizer**  is the same as that of  [tf.keras.mixed_precision.LossScaleOptimizer](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer).

Replace the occurrences of  **tf.keras.mixed_precision.LossScaleOptimizer**  in your script with  **npu.train.optimizer.NpuLossScaleOptimizer**  directly. If your script uses a different form of LossScaleOptimizer, import it to  **tf.keras.mixed_precision.LossScaleOptimizer**  and validate the functionality and quality before replacement.

## Setting the Number of Iterations Offloaded to NPU

You only need to set the  **npu loop size**, number of offload iterations on an NPU at the porting point. There are two methods:

- Use the environment variable  **NPU_LOOP_SIZE**  to set this parameter:

    ```bash
    export NPU_LOOP_SIZE=32
    ```

    This variable should be set before the  **import npu_device**  operation.

- Call the  [npu.set_npu_loop_size](../../apiref/npu-set_npu_loop_size.md)  API in your training script to set this parameter. Therefore, you need to understand the meaning of  **npu loop size**.

Let's first take an overview of the performance defects in the native TF2 workflow. Taking GPU as an example, the workflow timeline for standard GPU training is shown in the figure below. The user controls the script to run ten training rounds. One training step is executed on the GPU in each round. Once the training step finishes, execution control switches back to the Python side. The user verifies whether the current round count hits 10; if not, the next round is triggered, and this loop continues until all ten rounds are fully executed.

![](../figures/set_iterations_number_1.png)

As shown in the sequence diagram, both the CPU and GPU work intermittently, which brings the following performance issues:

- The Python interpreter has extra overhead and unpredictable time consumption. The gaps between successive training steps cause performance black holes.
- It is possible to accelerate the preprocessing pipeline by leveraging the dataset prefetching function of TF2. However, the time spent on host to device \(H2D\) data transfer and CPU scheduling in every training epoch is inescapable.

In TF2, to avoid the extra overhead on the Python interpreter, you are advised to use operator While to implement the training loop \(which is not a policy exclusive to the NPU\). In this case, it is operator While, instead of the Python interpreter, that determines whether the specified number of steps is reached. Organize your training script as follows.

```python
@tf.function
def loop_train(iterator, steps):
    for i in tf.range(steps):
        train_step(next(iterator))
```

After the TF2 code is compiled, the training steps are nested in operator While. The following figure shows the new execution timing.

![](../figures/set_iterations_number_2.png)

With the iteration offload policy, the time consumed on the Python interpreter is transferred to the TF CPU, which is shorter and more predictable. However, this mode also brings two extra overheads:

- Time spent in H2D data transfer in preprocessing
- Time spent by the operator in determining whether the specified number of steps is reached

To achieve optimal performance, the NPU employs the following two techniques to eliminate these two overheads:

- An asynchronous preprocessing H2D thread is used to asynchronize preprocessing output transfer from NPU training, hiding H2D transfer time within the NPU training time.
- The number of offloaded iterations is specified by the user to avoid the operator's time consumption, which also indicates the number of asynchronous H2D data transfers.

Asynchronous data transfer indicates that the TF Adapter's preprocessing thread proactively sends training data to the NPU. The execution timing without iteration offload is as follows.

In this case, the time consumed by preprocessing H2D data transfer and CPU scheduling can be reduced to some extent. \(Data transfer is in progress when a training step is delivered.\)

![](../figures/set_iterations_number_3.png)

The NPU execution timing with iteration offload is as follows.

![](../figures/set_iterations_number_4.png)

You can obtain the following information from the figure above.

- After the script delivers a training job of 10 epochs to the NPU, there are no more interactions between the script and the Python interpreter until the training is complete.
- The variation of preprocessing time consumption can be offset by the preprocessing time consumption leading NPU computation in the previous training step, increasing the tolerance to the preprocessing performance fluctuation.

To minimize the time consumed by training computation and maximize the performance gains, iteration offload to the NPU and asynchronous transfer of preprocessed data are used. However, your training job must meet the following requirement:

As the preprocessing thread is asynchronous with NPU computation, iteration offload requires a mechanism to notify the NPU of the number of currently offloaded iterations. The simplest approach is to set the NPU loop size.

See the following example:

```python
@tf.function
def loop_train(iterator, steps):
    for i in tf.range(steps):
        train_step(next(iterator))
```

If you wish to train 100 steps each time  **loop_train**  is called, you can set the NPU loop size in either of the following ways:

- Set the  **NPU_LOOP_SIZE**  environment variable before starting training.

    ```bash
    export NPU_LOOP_SIZE=100
    ```

- Insert a call to  [npu.set_npu_loop_size](../../apiref/npu-set_npu_loop_size.md)  \(pass  **100**  as the loop size\) before the  **loop_train**  call in your Python script.

    ```python
    npu.set_npu_loop_size(100)
    loop_train(train_iter, tf.constant(100))
    ```

Alternatively, call  **npu.set_npu_loop_size**  during training to change the number of steps offloaded to the NPU in each loop. For example, for a 100-step training job, if you want the NPU to execute 30 steps per loop, the remaining 91 to 100 steps \(10 steps\) will be less than the NPU loop size. Therefore, you can call  [npu.set_npu_loop_size](../../apiref/npu-set_npu_loop_size.md)  to adjust the NPU loop size after completing 90 steps.

```python
remaining_steps = 100  # Number of remaining steps
base_loop_size = 30  # Benchmark NPU loop size
npu.set_npu_loop_size(base_loop_size)
while remaining_steps >= base_loop_size:  # Offload based on benchmark loop size until the number of remaining steps is less than one loop.
    loop_train(train_iterator, tf.constant(base_loop_size))
    remaining_steps -= base_loop_size
if remaining_steps > 0:  # Process the remainder steps as a smaller NPU loop size.
    npu.set_npu_loop_size(remaining_steps)
    loop_train(train_iterator, tf.constant(remaining_steps))
```

## Distributed Training Script Adaptation \(Single Device\)

The following figure shows the distributed deployment topology on NPUs. Each TensorFlow process occupies and manages one dedicated NPU device. Inter-process synchronization is performed via the collective communication APIs provided by CANN. The only difference from single NPU training is that distributed NPU training involves collective communication.

![](../figures/distributed_deploy_mode.png)

The TF Adapter considers a single-device setup as a distributed NPU setup containing only one worker, unifying the training script for single NPU training and distributed NPU training.

Compared with single NPU training, distributed NPU training requires the following additional adaptation steps:

1. **Synchronizing initial values of variables between workers**

    In TF2 eager execution, variables are initialized immediately when a model is generated. It is important to synchronize the initial values of variables between the workers.

    When model building is complete, pass the variables to be synchronized to the  [npu.distribute.broadcast](../../apiref/npu-distribute-broadcast.md)  API call. You can call  **model.trainable_variables**  to obtain all the variables that need to be synchronized.

2. **Aggregating gradients between workers**

    Gradients generated on different workers at training time are aggregated to evaluate the training error.

    - If the original script computes and updates gradients in separate steps \(for example, by using  **tf.gradients**  and  **opt.apply_gradients**\), call the  [npu.distribute.all_reduce](../../apiref/npu-distribute-all_reduce.md)  API to aggregate the gradients. This API requires you to pass in the gradients to be aggregated across workers and specify the aggregation operation type \(typically mean reduction\).
    - If the original script uses a single API \(for example,  **minimize**  or  **model.fit**\) to compute and update gradients, call  [npu.distribute.npu_distributed_keras_optimizer_wrapper](../../apiref/npu-distribute-npu_distributed_keras_optimizer_wrapper.md)  to aggregate the gradients.

3. **Sharding dataset to workers**

    In distributed training, each worker uses different samples to better reflect the actual distribution of the training dataset. For example, to train a model using an 8-NPU cluster, a typical strategy is to shard elements 0–1/8 to the first NPU, elements 1/8–2/8 to the second NPU, ..., and elements 7/8–8/8 to the last NPU.

    - If a dataset is in  **tf.data.Dataset**  format, use the  [npu.distribute.shard_and_rebatch_dataset](../../apiref/npu-distribute-shard_and_rebatch_dataset.md)  API provided by the TF Adapter. This API call takes in the dataset to be sharded and the cluster's global batch size. Click  [here](https://www.tensorflow.org/guide/data?hl=en)  to find more about the dataset.

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

    - For a dataset from NumPy arrays, call related NumPy methods to shard the dataset and global batch. For example:

        ```python
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data(os.path.join(args.data_path, 'mnist.npz'))

        # Evenly divide the dataset based on the number of devices.
        x_trains = np.split(x_train, args.rank_size)
        # Obtain the dataset shard by device ID.
        x_train = x_trains[args.device_id]
        x_tests = np.split(x_test, args.rank_size)
        x_test = x_tests[args.device_id]
        # Shard the global batch.
        batch_size = args.batch_size // args.rank_size

        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        ```

## Using Training Startup Parameters Consistent with Single CPU Training

Modify the parameters related to distributed training in your startup script.

The current version requires you to start training in the single CPU form. That is, use the parameters for starting single-CPU training to start training on the  AI processor.

This will not alter the actual setup form of your script.

If your script has the distributed strategy parameter, set it to  **one_device**  \(corresponding to  **OneDeviceStrategy**\). Set the number of GPUs \(if configurable\) to 0.

This solution is used because:

- The training process is streamlined to the TF Adapter's perspective.
- The interference of the default distributed strategy of the original script is shielded.
