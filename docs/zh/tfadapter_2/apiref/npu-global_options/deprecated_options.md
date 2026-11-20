# 后续版本废弃配置

## op_select_implmode

> 该参数后续版本将废弃，建议使用[op_precision_mode](./performance_tuning.md#op_precision_mode)参数。

NPU内置算子有高精度和高性能实现方式，用户可以通过该参数配置模型编译时选择哪种算子。

- high_precision：表示算子选择高精度实现。高精度实现算子是指在fp16输的情况下，通过泰勒展开/牛顿迭代等手段进一步提升算子的精度。
- high_performance：表示算子选择高性能实现。高性能实现算子是指在fp16入的情况下，不影响网络精度前提的最优性能实现。

默认值为None，代表不使能此配置。

配置示例：

```python
npu.global_options().op_select_implmode="high_precision"
```

## optypelist_for_implmode

> 该参数后续版本将废弃，建议使用[op_precision_mode](./performance_tuning.md#op_precision_mode)参数。

列举算子optype的列表，该列表中的算子使用op_select_implmode参数指定的模式，当前支持的算子为Pooling、SoftmaxV2、LRN、ROIAlign，多个算子以“,”分隔。

该参数需要与op_select_implmode参数配合使用，配置示例：

```python
npu.global_options().op_select_implmode="high_precision"
npu.global_options().optypelist_for_implmode="Pooling,SoftmaxV2"
```

默认值为None，代表不使能此配置。

## variable_format_optimize

> 该参数为功能调试开关，后续版本将废弃，不建议用户使用。

是否开启变量格式优化。

- True：开启。
- False：关闭。

为了提高训练效率，在网络执行的变量初始化过程中，将变量转换成更适合在NPU上行的数据格式。但在用户特殊要求场景下，可以选择关闭该功能开关。

默认值为None，代表不使能此配置。

配置示例：

```python
npu.global_options().variable_format_optimize=True
```

## op_debug_level

> 该参数后续版本将废弃，建议使用[op_debug_config](./debugging.md#op_debug_config)。

算子debug功能开关。

- 0：不开启算子debug功能。
- 1：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成BE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o.json文件），用于后续工具进行AI Core Error问题定位。

  注意：Ascend 950PR&950DT系列产品不会生成TBE指令映射文件。

- 2：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成BE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o.json文件），并关闭ccec编译器的编译优化开关且打开ccec调试功能（ccec编器选项设置为-O0-g），用于后续工具进行AI Core Error问题定位。

  注意：Ascend 950PR&950DT系列产品不会生成TBE指令映射文件。

- 3：不开启算子debug功能，且在训练脚本执行目录下的kernel_meta文件夹中留.o和.json文件。
- 4：不开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中保留.o（算子二进制文件）和.json文件（算子描述文件），生成TBE指令映射文（算子cce文件\*.cce）和UB融合计算描述文件（\{$kernel_name\}_compute.son）。

  注意：

  - Ascend 950PR&950DT系列产品不会生成TBE指令映射文件和UB融合计算描述文件。
  - 当该参数取值为0时，同时又配置了“op_debug_config”参数，则训练执行时，仍会在当前执行路径下生成算子编译目录kernel_meta，目录中生成的内容以“op_debug_config”配置为准。
  - 训练执行时，建议配置为0或3。如果需要进行问题定位，再选择调试开关选项1和2，是因为加入了调试功能后，会导致网络性能下降。
  - 配置为2（即开启ccec编译选项）的场景下，不能与“op_debug_config”中的oom”同时使用，会导致AI Core Error报错，报错信息示例如下。

    ```text
    ...there is an aivec error exception, core id is 49, error ode = 0x4 ...
    ```

  - 配置为2（即开启ccec编译选项）的场景下，会增大算子Kernel（\*.o文件）大小。动态shape场景下，由于算子编译时会遍历可能存在的所有场景，最终可会导致由于算子Kernel文件过大而无法进行编译的情况，此种场景下，建议不配置为2。

    由于算子kernel文件过大而无法编译的日志显示如下：

    ```text
    message:link error ld.lld: error: InputSection too large for ange extension thunk ./kernel_meta_xxxxx.o:(xxxx)
    ```

  - 当该参数取值不为0时，可通过“debug_dir”参数指定调试相关过程文件的存放径。
  - 该参数取值为0，同时设置了NPU_COLLECT_PATH环境变量的场景，执行命令当路径下仍旧会生成算子编译目录kernel_meta；若设置了SCEND_WORK_PATH环境变量，则在该环境变量指定路径下生成kernel_meta。关环境变量的详细说明，可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。
  - debug功能开关打开场景下，若模型中含有如下通算融合算子，算子编译目录ernel_meta中，不会生成下述算子的\*.o、\*.json、\*.cce文件。

    ```text
    MatMulAllReduce
    MatMulAllReduceAddRmsNorm
    AllGatherMatMul
    MatMulReduceScatter
    AlltoAllAllGatherBatchMatMul
    BatchMatMulReduceScatterAlltoAll
    ```

默认值为None，代表不使能此配置。

配置示例：

```python
npu.global_options().op_debug_level=0
```

## graph_memory_max_size

历史版本，该参数用于指定网络静态内存和最大动态内存的大小；当前版本，该参数不再生效。系统会根据网络使用的实际内存大小动态申请。

## variable_memory_max_size

历史版本，该参数用于指定变量内存的大小；当前版本，该参数不再生效。系统会根据网络使用的实际内存大小动态申请。
