# Experiment Parameters

The experiment parameters are extended parameters for debugging and may be changed in later versions. Therefore, they cannot be used in production environments.

## graph_compiler_cache_dir

Drive cache directory for graph compilation. If this parameter is not empty, the drive cache function for graph compilation takes effect.

The graph compilation cache function supports drive persistence of graph compilation results. When graph compilation is performed again, the compilation results cached on the drive can be directly loaded to reduce the graph compilation duration.

Note:

- The configured cache directory must exist. Otherwise, the compilation fails.
- During graph compilation, the cache file is determined based on the value of this parameter. If the cache file does not exist, the cache is saved. If the cache file exists, the existing cache is directly loaded.
- After a graph is changed, the original cache file is unavailable. You need to manually delete the cache file from the cache directory or rebuild and generate a cache file.
- The cache does not ensure cross-version compatibility. If the version is upgraded, clear the cache directory and rebuild and generate the cache.
- This function does not support models with resource operators.

Example:

```python
custom_op.parameter_map["graph_compiler_cache_dir"].s = tf.compat.as_bytes("/root/build_cache_dir")
```

## jit_compile

Determines whether to compile the operator online or use the compiled operator binary file.

- auto: For a static shape network, compile the operator online. For a dynamic shape network, search for the compiled operator binary in the system first. If the corresponding binary file is not available, compile the operator.
- true: Operators are compiled online. The system performs fusion and tuning based on the obtained graph information to get better performing operators.
- false: The compiled operator binary file in the system is preferentially searched. If the file can be found, operators are not compiled anymore, which produces better compilation performance. If the file cannot be found, operators will be compiled.

> [!NOTE]
> This option is used only for networks of large recommendation models.

Example:

```python
custom_op.parameter_map["jit_compile"].s = tf.compat.as_bytes( "auto")
```

## shape_generalization_mode

When jit_compile is set to true (online operator compilation), use this parameter to configure the shape generalization mode.

- STRICT (default): Uses the shape of the current iteration as is, without any generalization.
- FULL: Generalizes all axes to -1 if the shape changes between iterations.
- ADAPTIVE: Generalizes only the shape of the changed axis to -1 if the shape changes between iterations. The newly generalized axis triggers model recompilation, which may cause the model to be compiled multiple times under this configuration.

> [!NOTE]NOTE
> When [compile_dynamic_mode](../../apiref/session_config/dynamic_shape.md#compile_dynamic_mode) is set to True, the first iteration generalizes all input shapes to -1, and the shape_generalization_mode setting does not take effect.

Example:

```python
custom_op.parameter_map["shape_generalization_mode"].s = tf.compat.as_bytes( "FULL")
```

## experimental_accelerate_train_mode

If training takes more than one hour, you can trigger training acceleration to improve training performance by configuring this option.

Based on the configured acceleration type, acceleration trigger mode, and the proportion of low-precision training processes, the software compiles and runs the corresponding proportion of training processes with reduced precision, while the remaining processes are compiled and run at their original precision.

The value of this option is a string with three fields separated by vertical bars (|), for example, fast|step|0.9.

- The first field indicates the acceleration type, which can be fast or fast1.
  - fast: Indicates that the compilation is performed based on the float16 data type during precision reduction.
  - fast1: that the compilation is performed based on the bf16 data type during precision reduction.
- The second field supports two values: step and loss, indicating that the entire training process is divided into low-precision training and high-precision training based on the step or loss value, respectively.
- The third field indicates the proportion of the training process that runs in low precision, relative to either the total step count or the total loss range.
  - When the second field is step, its value ranges from 0.2 to 0.9. The default value is 0.9.
  - When the second field is loss, its value ranges from 1.01 to 1.5. The default value is 1.05.

Example:

Acceleration triggered by step:

```python
custom_op.parameter_map["experimental_accelerate_train_mode"].s =
tf.compat.as_bytes("fast|step|0.9")
```

Acceleration triggered by loss:

```python
custom_op.parameter_map["experimental_accelerate_train_mode"].s =
tf.compat.as_bytes("fast|loss|1.05")
```

NOTE:

1. To use this option for training acceleration, ensure that the network script can converge properly.
2. For training scripts with short execution time, enabling this option may not bring positive end-to-end performance gains.
3. The function of this option is related to the precision mode configured in the network script:
   - When precision_mode is used to configure the precision mode, this option can be enabled only when precision_mode is set to allow_fp32_to_fp16, must_keep_origin_dtype, or none.
   - When precision_mode_v2 is used to configure the precision mode, this option can be enabled only when precision_mode_v2 is set to origin or none.
4. This option is related to the number of small loops. Enabling small loops may result in inaccurate splitting of the training process based on the specified steps or loss value, which may ultimately affect the loss and accuracy.
5. When this option is enabled, you need to modify the network script based on the following rules:
   - If the entire training process is split by step, you need to set the STEP_NOW and TOTAL_STEP environment variables to notify the bottom layer of the step value and the total number of steps of each run.
   - If the entire training process is split by loss, you need to set the LOSS_NOW and TARGET_LOSS environment variables to notify the bottom layer of the loss value and target loss of each run.

The following is an example of modifying the network script in step splitting mode:

```python
# Set the initial value of STEP_NOW to 0.
os.environ['STEP_NOW'] =  "0"
# Set TOTAL_STEP to the total number of steps.
os.environ['TOTAL_STEP'] =  str(epoch)
for i in range(epoch):
    # Start training.
    _, step = sess.run([train_op, global_step])
    # Update the value of STEP_NOW to the current step.
    os.environ['STEP_NOW'] =  str(step)
```

The following is an example of modifying the network script in loss splitting mode:

```text
# Set the initial value of LOSS_NOW to the initial loss value of the network. The following value is for reference only.
os.environ['LOSS_NOW'] =  "7.0"
# Set the value of TARGET_LOSS to the target loss value. The following value is for reference only.
os.environ['TARGET_LOSS'] =  "3.0"
for i in range(epoch):
    # Start training.
    _, step = sess.run([train_op, global_step])
    # Update the value of LOSS_NOW to the loss value that is being executed.
    os.environ['LOSS_NOW'] =  str(loss)
```

## auto_multistream_parallel_mode

This option applies to static and dynamic shape graph scenarios. You can enable parallel execution of Cube and Vector operators to improve graph execution performance.

- **cv**, Parallel execution of Cube and Vector operators is enabled.
- **LoadBalance:n**, Load balancing algorithm that distributes all operators evenly across n streams for execution. Here, n represents the maximum number of streams, which must be a positive integer within the range [1, 64]. If n exceeds the number of available cores, performance may degrade.
- **MainStream:n**，Main stream algorithm that executes serial operators on the main stream, while other parallelizable operators are distributed across other streams. Here, n represents the maximum number of streams, which must be a positive integer within the range [1, 64]. If n exceeds the number of available cores, performance may degrade.
- The default value is empty, meaning Cube and Vector operators are executed serially.

> [!NOTE]NOTE
>
> - This option is used only for recommendation networks.
> - To use this function in dynamic shape multi-stream mode, you need to first enable dynamic shape multi-stream by setting the environment variable ENABLE_DYNAMIC_SHAPE_MULTI_STREAM, and then configure this option. For details about environment variables, see [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Example:

```python
custom_op.parameter_map["auto_multistream_parallel_mode"].s =
tf.compat.as_bytes("cv")
```
