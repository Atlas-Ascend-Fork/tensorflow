# 后续版本废弃配置

## enable_data_pre_proc

> 当前版本该参数不再生效，系统会自适应判断GetNext算子是否下沉到NPU执行。

性能调优相关配置，用于指定GetNext算子是否下沉到NPU侧执行，GetNext算子下沉是使能训练迭代循环下沉的必要条件。

- True（默认值）：下沉。GetNext算子下沉的前提是必须使用TensorFlow Dataset方式读数据。
- False：不下沉。

配置示例：

```python
config = NPURunConfig(enable_data_pre_proc=True)
```

## variable_format_optimize

> 该参数为功能调试开关，后续版本将废弃，不建议用户使用。

性能调优相关配置，是否开启变量格式优化。

- True：开启。
- False：关闭。

为了提高训练效率，在网络执行的变量初始化过程中，将变量转换成更适合在AI处理器上运行的数据格式。但在用户特殊要求场景下，可以选择关闭该功能开关。

默认值为空，代表不使能此配置。

配置示例：

```python
config = NPURunConfig(variable_format_optimize=True)
```

## op_debug_level

> 该参数后续版本将废弃，建议使用[op_debug_config](./debugging.md#op_debug_config)参数。

算子debug功能开关。

- 0：不开启算子debug功能。
- 1：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o和.json文件），用于后续工具进行AI Core Error问题定位。

  注意：Ascend 950PR&950DT系列产品不会生成TBE指定映射文件。

- 2：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o和.json文件），并关闭ccec编译器的编译优化开关且打开ccec调试功能（ccec编译器选项设置为-O0-g），用于后续工具进行AI Core Error问题定位。

  注意：Ascend 950PR&950DT系列产品不会生成TBE指定映射文件。

- 3：不开启算子debug功能，且在训练脚本执行目录下的kernel_meta文件夹中保留.o和.json文件。
- 4：不开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中**保留**.o（算子二进制文件）和.json文件（算子描述文件），生成TBE指令映射文件（算子cce文件\*.cce）和UB融合计算描述文件（\{$kernel_name\}_compute.json）。

  Ascend 950PR&950DT系列产品不会生成TBE指定映射文件和UB融合计算描述文件。

**注意：**

- 当该参数取值为0时，同时又配置了“op_debug_config”参数，则训练执行时，仍会在当前执行路径下生成算子编译目录kernel_meta，目录中生成的内容以“op_debug_config”配置为准。
- 训练执行时，建议配置为0或3。如果需要进行问题定位，再选择调试开关选项1和2，是因为加入了调试功能后，会导致网络性能下降。
- 配置为2（即开启ccec编译选项）的场景下，不能与“op_debug_config”中的“oom”同时使用，会导致AI Core Error报错，报错信息示例如下。

  ```text
  ...there is an aivec error exception, core id is 49, error code = 0x4 ...
  ```

- 配置为2（即开启ccec编译选项）的场景下，会增大算子Kernel（\*.o文件）的大小。动态shape场景下，由于算子编译时会遍历可能存在的所有场景，最终可能会导致由于算子Kernel文件过大而无法进行编译的情况，此种场景下，建议不要配置为2。
  
  由于算子kernel文件过大而无法编译的日志显示如下：

  ```text
  message:link error ld.lld: error: InputSection too large for range extension thunk ./kernel_meta_xxxxx.o:(xxxx)
  ```

- 当该参数取值不为0时，可通过“debug_dir”参数指定调试相关过程文件的存放路径。
- 该参数取值为0，同时设置了NPU_COLLECT_PATH环境变量的场景，执行命令当前路径下仍旧会生成算子编译目录kernel_meta；若设置了ASCEND_WORK_PATH环境变量，则在该环境变量指定路径下生成kernel_meta。关于环境变量的详细说明，可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。
- debug功能开关打开场景下，若模型中含有如下通算融合算子，算子编译目录kernel_meta中，不会生成下述算子的\*.o、\*.json、\*.cce文件。

  ```text
  MatMulAllReduce
  MatMulAllReduceAddRmsNorm
  AllGatherMatMul
  MatMulReduceScatter
  AlltoAllAllGatherBatchMatMul
  BatchMatMulReduceScatterAlltoAll
  ```

默认值为空，代表不使能此配置。

配置示例：

```python
config = NPURunConfig(op_debug_level=1)
```

## op_select_implmode

> 该参数后续版本将废弃，建议使用[op_precision_mode](./performance_tuning.md#op_precision_mode)参数。

NPU内置算子有高精度和高性能实现方式，用户可以通过该参数配置模型编译时选择哪种算子。

- high_precision：表示算子选择高精度实现。高精度实现算子是指在fp16输入的情况下，通过泰勒展开/牛顿迭代等手段进一步提升算子的精度。
- high_performance：表示算子选择高性能实现。高性能实现算子是指在fp16输入的情况下，不影响网络精度前提的最优性能实现。

默认值为空，代表不使能此配置。

配置示例：

```python
config = NPURunConfig(op_select_implmode="high_precision")
```

## optypelist_for_implmode

> 该参数后续版本将废弃，建议使用[op_precision_mode](./performance_tuning.md#op_precision_mode)参数。

列举算子optype的列表，该列表中的算子使用op_select_implmode参数指定的模式，当前支持的算子为Pooling、SoftmaxV2、LRN、ROIAlign，多个算子以“,”分隔。

`optypelist_for_implmode`参数需要与op_select_implmode参数配合使用，配置示例：

```python
config = NPURunConfig(
    op_select_implmode="high_precision",
    optypelist_for_implmode="Pooling,SoftmaxV2")
```

默认值为空，代表不使能此配置。

## dynamic_input

> 该参数后续版本将废弃，不建议开发者使用。当前版本已默认支持动态shape网络，可参见[jit_compile](./experiment_options.md#jit_compile)参数。

当前网络的输入是否为动态输入，取值包括：

- True：动态输入。
- False（默认值）：固定输入。

配置示例：

```python
config = NPURunConfig(dynamic_input=True)
```

## dynamic_graph_execute_mode

> 该参数后续版本将废弃，不建议开发者使用。当前版本已默认支持动态shape网络，可参见[jit_compile](./experiment_options.md#jit_compile)参数。

对于动态输入场景，需要通过该参数设置执行模式，即dynamic_input为True时该参数生效。取值为：

dynamic_execute：动态图编译模式。该模式下获取dynamic_inputs_shape_range中配置的shape范围进行编译。

配置示例：

```python
config = NPURunConfig(dynamic_graph_execute_mode="dynamic_execute")
```

## dynamic_inputs_shape_range

> 该参数后续版本将废弃，不建议开发者使用。当前版本已默认支持动态shape网络，可参见[jit_compile](./experiment_options.md#jit_compile)参数。

动态输入的shape范围。例如全图有3个输入，两个为dataset输入，一个为placeholder输入，则配置示例为：

```python
config = NPURunConfig(dynamic_inputs_shape_range="getnext:[128 ,3~5, 2~128, -1],[64 ,3~5, 2~128, -1];data:[128 ,3~5, 2~128, -1]")
```

使用注意事项：

- dataset输入固定标识为“getnext”，placeholder输入固定标识为“data”，不允许用其他表示。
- 动态维度有shape范围的用波浪号“\~”表示，固定维度用固定数字表示，无限定范围的用-1表示。
- 对于多输入场景，例如有三个dataset输入时，如果只有第二个第三个输入具有shape范围，第一个输入为固定输入时，仍需要将固定输入shape填入：

  ```python
  config = NPURunConfig(dynamic_inputs_shape_range="getnext:[3,3,4,10],[-1,3,2~1000,-1],[-1,-1,-1,-1]")
  ```

- 对于标量输入，也需要填入shape范围，表示方法为：\[\]，"\[\]"前不允许有空格。
- 若网络中有多个getnext输入，或者多个data输入，需要分别保持顺序关系，例如：

  - 若网络中有多个dataset输入：

    ```python
    def func(x):
        x = x + 1
        y = x + 2
        return x,y
    dataset = tf.data.Dataset.range(min_size, max_size)
    dataset = dataset.map(func)
    ```

    网络的第一个输入是x（假设shape range为：\[3\~5\]），第二个输入是y（假设shape range为：\[3\~6\]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range ="getnext:[3~5],[3~6]")
    ```

  - 若网络中有多个placeholder输入：

    如果不指定placeholder的name，例：

    ```python
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    ```

    placeholder的顺序和脚本中定义的位置一致，即网络的第一个输入是x（假设shape range为：\[3\~5\]），第二个输入是y（shape range为：\[3\~6\]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range= "data:[3~5],[3~6]")
    ```

    如果指定了placeholder的name，例：

    ```python
    x = tf.placeholder(tf.int32, name='b')
    y = tf.placeholder(tf.int32, name='a')
    ```

    则网络输入的顺序按name的字母序排序，即

    即网络的第一个输入是y（假设shape range为：\[3\~6\]），第二个输入是x（shape range为：\[3\~5\]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range = "data:[3~6],[3~5]")
    ```

    **注意：**
    - 当存在不同输入shape的子图时，由于dynamic_inputs_shape_range是针对于单张图的配置属性，因此可能会导致执行异常，建议使用[set_graph_exec_config](../../npu_util/set_graph_exec_config.md)以支持动态输入场景。
    - 若网络脚本中未指定placeholder的name，则placeholder会按照如下格式命名：

      xxx_0, xxx_1, xxx_2, ……
      
      其中下划线后为placeholder在网络脚本中的定义顺序索引，placeholder会按照此索引的字母顺序进行排布，所以当placeholder的个数大于10时，则排序为“xxx_0 -\> xxx_10 -\> xxx_2 -\> xxx_3”，网络脚本中定义索引为10的placeholder排在了索引为2的placeholder前面，导致定义的shape range与实际输入的placeholder不匹配。

      为避免此问题，当placeholder的输入个数大于10时，建议在网络脚本中指定placeholder的name，则placeholder会以指定的name进行命名，实现shape range与placeholder name的关联。
    - 该参数不允许与dynamic_dims同时使用，若同时使用，dynamic_dims优先级更高，此参数不生效。

## graph_memory_max_size

- 历史版本，该参数用于指定网络静态内存和最大动态内存的大小。
- 当前版本，该参数不再生效。系统会根据网络使用的实际内存大小动态申请。

## variable_memory_max_size

- 历史版本，该参数用于指定变量内存的大小。
- 当前版本，该参数不再生效。系统会根据网络使用的实际内存大小动态申请。
