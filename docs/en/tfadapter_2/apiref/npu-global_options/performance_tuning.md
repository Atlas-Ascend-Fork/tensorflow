# Performance Tuning

## hcom_parallel

Enable for the AllReduce gradient update and forward and backward propagation in parallel.

- True (default): enabled.
- False: disabled.

Example:

```python
npu.global_options().hcom_parallel=True
```

For a small network (for example, ResNet-18), you are advised to set this option to False.

## enable_small_channel

Whether to enable small channel optimization. If it is enabled, performance benefits are generated at the convolutional layers with channel size ≤ 4.

- 0: disabled. This function is disabled by default in the training scenario (graph_run_mode is 1). You are advised not to enable this function in the training scenario.
- 1: enabled. This is the default option that cannot be modified for the online inference scenario (graph_run_mode is 0).
  > [!NOTE]NOTE
  > After this function is enabled, performance gains can be obtained only on the ResNet-50, ResNet-101, and ResNet-152 networks. For other networks, the performance may deteriorate.

Example:

```python
npu.global_options().enable_small_channel=1
```

## op_precision_mode

High-precision or high-performance mode of an operator. You can pass a custom mode configuration file op_precision.ini to set different modes for operators.

You can set this option by operator type (low priority) or node name (high priority). Example:

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

You can view the precision and performance mode supported by an operator in the opp/built-in/op_impl/ai_core/tbe/impl_mode/all_ops_impl_mode.ini file of the CANN component directory.

This option is mutually exclusive with op_select_implmode and optypelist_for_implmode. If they are all specified, op_precision_mode takes precedence.

Generally, you do not need to set this option. It is used if you need to adjust the precision of a specific operator using the configuration .ini file in the case that you fail to obtain optimal network performance or accuracy in the high-performance or high-precision mode.

Example:

 ```python
 npu.global_options().op_precision_mode="/home/test/op_precision.ni"
 ```

## stream_max_parallel_num

This option applies only to neural machine translation (NMT) networks.

Degree of parallelism of AI CPU and AI Core engines for parallel execution of AI CPU and AI Core operators.

DNN_VM_AICPU is the name of the AI CPU engine. In this example, the number of concurrent tasks on the AI CPU engine is 10.

AIcoreEngine is the name of the AI Core engine. In this example, the number of concurrent tasks on the AI Core engine is 1.

The parallelism of the AICPU/AICORE engine defaults to 1, with a valid range of [1, 13].

Example:

```python
npu.global_options().stream_max_parallel_num="DNN_VM_AICPU:10,IcoreEngine:1"
```

## is_tailing_optimization

This option applies only to Bidirectional Encoder Representations from Transformers (BERT) networks.

Whether to enable communication tailing optimization to improve training performance in distributed training scenarios. By changing a computation dependency relationship, a computation operation that does not depend on the last AR (gradient aggregation fragment) is scheduled to be performed in parallel with the last AR, to optimize communication tailing. 

Values:

- True: enabled.
- False (default): disabled.

Example:

```python
npu.global_options().is_tailing_optimization=True
```

## enable_scope_fusion_passes

Fusion pattern (or fusion patterns separated by commas) to take effect at build time.

Scope fusion patterns (either built-in or custom) are classified into the following two types:

- General scope fusion patterns: applicable to all networks. They are enabled by default and cannot be manually disabled.
- Non-general scope fusion patterns: applicable to specific networks. By default, they are disabled. You can use enable_scope_fusion_passes to enable selected fusion patterns.

Example:

```python
npu.global_options().nable_scope_fusion_passes="ScopeLayerNormPass,ScopeClipBoxesPass"
```
