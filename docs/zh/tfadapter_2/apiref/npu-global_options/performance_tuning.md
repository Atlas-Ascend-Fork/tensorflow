# 性能调优

## hcom_parallel

是否启用Allreduce梯度更新和前后向并行执行

- True（默认值）：开启Allreduce并行。
- False：关闭Allreduce并行。

配置示例：

```python
npu.global_options().hcom_parallel=True
```

针对小网络（例如：ResNet18），建议配置为False。

## enable_small_channel

是否使能small channel的优化，使能后在channel<=4的卷积层会有性能收益。

- 0：关闭。训练（graph_run_mode为1）场景下默认关闭，且训练场景下不建议用户开启。
- 1：使能。在线推理（graph_run_mode为0）场景下不支持用户配置，默认使能。
  > [!NOTE]说明
  > 该参数使能后，当前只在ResNet50、ResNet101、ResNet152网络模型能获得性能收益。其他网络模型性能可能会下降，用户根据实际情况决定是否使能该参数。

配置示例：

```python
npu.global_options().enable_small_channel=1
```

## op_precision_mode

设置具体某个算子的高精度或高性能模式，通过该参数传入自定义的模式配置文件op_precision.ini，可以为不同的算子设置不同的模式。

 支持按照算子类型或者按照节点名称设置，按节点名称设置的优先级高于算子类，样例如下：

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

- high_precision：表示高精度。
- high_performance：表示高性能。
- enable_float_32_execution：算子内部处理时使用FP32数据类型功能，该景下FP32数据类型不会自动转换为HF32数据类型；若使用HF32计算，精度损失超预期时，可启用该配置，指定部分算子内部计算时使用FP32，保持精度。

   **该选项仅在以下产品支持：**

   Ascend 950PR&950DT系列产品

   Atlas A3系列产品

   Atlas A2系列产品

- enable_hi_float_32_execution：算子内部处理时使用HF32数据类型功能，能后，FP32数据类型自动转换为HF32数据类型；该配置可以降低数据所占空间大，实现性能提升。**当前版本暂不支持此配置。**
- support_out_of_bound_index：表示对gather、scatter和segment类算子indices输入进行越界校验，校验会降低算子的执行性能。
- keep_fp16：算子内部处理时使用FP16数据类型功能，该场景下FP16数据类型会自动转换为FP32数据类型；若使用FP32计算时性能不满足预期，同时精度要求高情况下，可以选择keep_fp16模式，**牺牲精度提升性能，不建议使用该低精模式**。
- super_performance：表示超高性能，和高性能相比，在算法计算公式上进行优化。

 具体某个算子支持配置的精度/性能模式取值，可通过CANN软件安装后文件存储路径“opp/built-in/op_impl/ai_core/tbe/impl_mode/all_ops_impl_mode.ni”文件查看。

 该参数不能与op_select_implmode、optypelist_for_implmode参数同时使用，三个参数同时配置，则只有op_precision_mode参数指定的模式生效。

 一般场景下该参数无需配置。若使用高性能或者高精度模式，网络性能或者精度不最优，则可以使用该参数，通过配置ini文件调整某个具体算子的精度模式。

 配置示例：

 ```python
 npu.global_options().op_precision_mode="/home/test/op_precision.ni"
 ```

## stream_max_parallel_num

此参数仅适用于NMT网络。

指定AICPU/AICORE引擎的并行度，从而实现AICPU/AICORE算子间的并行执行。

DNN_VM_AICPU为AICPU引擎名称，本示例指定了AICPU引擎的并发数为10。

AIcoreEngine为AICORE引擎名称，本示例指定了AICORE引擎的并发数为1。

AICPU/AICORE引擎的并行度默认为1，取值范围为：\[1,13\]。

配置示例：

```python
npu.global_options().stream_max_parallel_num="DNN_VM_AICPU:10,AIcoreEngine:1"
```

## is_tailing_optimization

此参数仅适用于Bert网络。

分布式训练场景下，是否开启通信拖尾优化，用于提升训练性能。通信拖尾优即，通过计算依赖关系的改变，将不依赖于最后一个AR（梯度聚合分片）的计算作调度到和最后一个AR并行进行，以达到优化通信拖尾时间的目的。

取值：

- True。
- False：默认为False。

配置示例：

```python
npu.global_options().is_tailing_optimization=True
```

## enable_scope_fusion_passes

指定编译时需要生效的融合规则列表。此处传入注册的融合规则名称，允许传入多个，用“,”隔开。

无论是内置还是用户自定义的Scope融合规则，都分为如下两类：

- 通用融合规则（General）：各网络通用的Scope融合规则；默认生效，不支持户指定失效。
- 定制化融合规则（Non-General）：特定网络适用的Scope融合规则；默认不效，用户可以通过enable_scope_fusion_passes指定生效的融合规则列表。

配置示例：

```python
npu.global_options().nable_scope_fusion_passes="ScopeLayerNormPass,ScopeClipBoxesPass"
```
