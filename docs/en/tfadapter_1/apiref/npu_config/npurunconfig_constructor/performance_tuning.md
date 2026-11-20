# Performance Tuning

## Basic Configuration

### iterations_per_loop

Number of iterations per training loop performed on the NPU per sess.run() call. Defaults to 1. The total number of training iterations per loop must be an integer multiple of the value of iterations_per_loop. Training is performed according to the specified number of iterations per loop (iterations_per_loop) on the NPU and then the result is returned to the host. This parameter can save unnecessary interactions between the host and device and reduce the training time consumption.

In mixed compute mode (with mix_compile_mode set to True), iterations_per_loop must be set to 1.

Note: When iterations_per_loop is set to a value greater than 1, the total number of training iterations set by the user may be different from the actual total number of iterations due to issues such as loop offloading and loss scale overflow.

Example:

```python
config = NPURunConfig(iterations_per_loop=1000)
```

## Advanced Setting

### hcom_parallel

Enables AllReduce gradient update and forward and backward propagation in parallel during distributed training.

- True (default): enabled.
- False: disabled.

For a small network (for example, ResNet-18), you are advised to set this parameter to False.

Example:

```python
config = NPURunConfig(hcom_parallel=True)
```

### op_precision_mode

High-precision or high-performance mode of an operator. You can pass a custom mode configuration file op_precision.ini to set different modes for operators.

You can set this parameter by operator type (low priority) or node name (high priority). Example:[ByOpType]

```text
[ByOpType]
optype1=high_precision
optype2=high_performance
optype3=enable_hi_float_32_execution
optype4=support_out_of_bound_index
[ByNodeName]
nodename1=high_precision
nodename2=high_performance
nodename3=enable_hi_float_32_execution
nodename4=support_out_of_bound_index
```

- high_precision: high precision.
- high_performance: high performance.
- enable_float_32_execution: The FP32 data type is used for internal processing of operators. In this scenario, the FP32 data type is not automatically converted to the HF32 data type. If you are using the HF32 data type for computation and find that the accuracy drop exceeds your expectation, you can enable this configuration to specify the use of FP32 for internal computation of certain operators in order to maintain accuracy.

This option is supported only by the following products:

Ascend 950PR&950DT products

Atlas A3 products

Atlas A2 products

- enable_hi_float_32_execution: The HF32 data type is used for internal processing of operators. After this configuration is enabled, the FP32 data type is automatically converted to the HF32 data type. This configuration can reduce the space occupied by data and improve performance. This configuration is not supported in the current version.
- support_out_of_bound_index: The out-of-bounds verification is performed on the indices of the gather, scatter, and segment operators. The verification deteriorates the operator execution performance.
- keep_fp16: The FP16 data type is used for internal processing of operators. In this scenario, the FP16 data type is not automatically converted to the FP32 data type. If the performance of FP32 computation does not meet the expectation and high precision is not required, you can select the keep_fp16 mode. This low-precision mode sacrifices the precision for improving the performance, which is not recommended.
- super_performance: ultra-high performance. Compared with high performance, the algorithm calculation formula is optimized.

You can view the precision or performance mode supported by an operator in the `opp/built-in/op_impl/ai_core/tbe/impl_mode/all_ops_impl_mode.ini` file in the file storage path with the CANN software installed.

This parameter is mutually exclusive with op_select_implmode and optypelist_for_implmode. If they are all specified, op_precision_mode takes precedence.

Generally, you do not need to set this parameter. It is used if you need to adjust the precision of a specific operator using the configuration .ini file in the case that you fail to obtain optimal network performance or accuracy in the high-performance or high-precision mode.

Example:

```python
config = NPURunConfig(op_precision_mode="/home/test/op_precision.ini")
```

## enable_scope_fusion_passes

Scope fusion pattern (or scope fusion patterns separated by commas) to take effect during build. Name of the registered fusion pattern. You can pass multiple names. Separate the names by commas (,).

Scope fusion patterns (either built-in or custom) are classified into the following two types:

- General: common scope fusion patterns applicable to all networks. They are enabled by default and cannot be manually invalidated.
- Non-general scope fusion patterns: applicable to specific networks. By default, they are disabled. You can use enable_scope_fusion_passes to enable selected fusion patterns.

Example:

```python
config = NPURunConfig(enable_scope_fusion_passes="ScopeLayerNormPass,ScopeClipBoxesPass")
```

### stream_max_parallel_num

This parameter applies only to NMT networks.

It specifies the degree of parallelism of AI CPU and AI Core engines for parallel execution of AI CPU and AI Core operators.

Example:

```python
config = NPURunConfig(stream_max_parallel_num="DNN_VM_AICPU:10,AIcoreEngine:1")
```

- DNN_VM_AICPU is the name of the AI CPU engine. In this example, the number of concurrent tasks on the AI CPU engine is 10.

- AIcoreEngine is the name of the AI Core engine. In this example, the number of concurrent tasks on the AI Core engine is 1.

- Defaults to 1. The value cannot exceed the maximum number of AI Cores.

### is_tailing_optimization

This parameter applies only to BERT networks.

Communication tailing optimization enable in distributed training scenarios to improve performance. By changing a computation dependency relationship, a computation operation that does not depend on the last AR (gradient aggregation fragment) is scheduled to be performed in parallel with the last AR, to optimize communication tailing.

- True: enabled.
- False (default): disabled.

This parameter must be used in pair with [NPUOptimizer Constructor](../../npu_optimizer/NPUOptimizer_constructor.md) and the value must be the same as that of is_tailing_optimization in [NPUOptimizer Constructor](../../npu_optimizer/NPUOptimizer_constructor.md).

Example:

```python
config = NPURunConfig(is_tailing_optimization=True)
```

### enable_small_channel

Whether to enable small channel optimization. If it is enabled, performance benefits are generated at the convolutional layers with channel size ≤ 4.

- 0: disabled. This function is disabled by default in the training scenario (graph_run_mode is 1). You are advised not to enable this function in the training scenario.
- 1: enabled. This is the default option that cannot be modified for the online inference scenario (graph_run_mode is 0).

> [!NOTE]NOTE
> After this function is included, performance benefits can be obtained on the ResNet50, ResNet101, and ResNet152 networks. For other network models, the performance may deteriorate.

Example:

```python
config = NPURunConfig(enable_small_channel=0)
```

### variable_placement

If the network weight is large, network execution may fail due to insufficient device memory. In this case, you can deploy the variable to the host to reduce the memory usage of the device.

- Device (default): The variable is deployed on the device.
- Host: The variable is deployed on the host.

Restrictions:

1. If this configuration option is set to Host, mixed computing must be enabled (mix_compile_mode = True).
2. If the training script contains APIs of TensorFlow V1 control flow operators, such as tf.case, tf.cond, and tf.while_loop, setting variable_placement to Host may cause the network execution to fail. To avoid this problem, add the following APIs to the training script to convert the control flow operators of TensorFlow V1 to V2 and enable resource variables:

   ```python
   tf.enable_control_flow_v2()
   tf.enable_resource_variables()
   ```

Example:

```python
config = NPURunConfig(variable_placement="Device")
```

### graph_max_parallel_model_num

In online inference scenarios, you can set this parameter to specify the maximum number of threads for parallel graph execution. If the value of this parameter is greater than 1, the corresponding number of threads are started for parallel graph execution, improving the overall graph pipeline efficiency.

The value must be an integer in the range of [1, INT32_MAX]. The default value is 1. INT32_MAX is the maximum value of the INT32 type, which is 2147483647.

Example:

```python
config = NPURunConfig(graph_max_parallel_model_num=4)
```
