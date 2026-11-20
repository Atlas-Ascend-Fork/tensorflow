# initialize_system

## 功能说明

一般执行训练不需要调用该接口，如果用户统计训练时间时不想包括GE初始化时间，可以使用该接口。使用集合通信接口时，需要先调用该接口进行集合通信初始化。

## 函数原型

```python
def initialize_system(name = None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 算子名称。 |

## 返回值

返回一个op，供用户通过`sess.run(op)`完成GE初始化。

## 约束说明

如果需要调用initialize_system接口，且训练执行时需要使能如下功能时，则必须在initialize_system起session时配置。

- **profiling_mode**：是否开启Profiling功能。
  - True：开启Profiling功能，从profiling_options读取Profiling的采集选项。
  - False：关闭Profiling功能，默认关闭。

- **profiling_options**：Profiling采集选项。

   支持的配置选项可参见环境变量[PROFILING_OPTIONS](https://gitcode.com/cann/oam-tools/blob/master/docs/zh/user_guide/env-vars/PROFILING_OPTIONS.md)。

- **enable_dump**：是否开启Data Dump功能。

  - True：开启Data Dump功能，从dump_path读取Dump文件保存路径，dump_path为None时会产生异常。
  - False（默认值）：关闭Data Dump功能。

    > [!NOTE]说明
    > - 不能同时开启Data Dump与溢出数据采集功能，即不同时将enable_dump和enable_dump_debug参数配置为“True”。
    > - 若“enable_dump/enable_dump_debug（二选一）”参数配置为“True”，同时“enable_exception_dump”配置为了“1”（即开启普通异常算子dump）。此时，针对动态shape网络，仅“enable_exception_dump”生效；针对静态shape网络，“enable_exception_dump”与“enable_dump/enable_dump_debug（二选一）”都生效。

- **dump_path**：Dump文件保存路径。enable_dump或enable_dump_debug为true时，该参数必须配置。

    该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径（相对执行命令行时的当前路径）。

  - 绝对路径配置以“/”开头，例如：/home/test/output。
  - 相对路径配置直接以目录名开始，例如：output。

- **dump_step**：指定采集哪些迭代的Data Dump数据。默认值：None，表示所有迭代都会产生dump数据。

    多个迭代用“|”分割，例如：0|5|10；也可以用“-”指定迭代范围，例如：0|3-5|10。

- **dump_mode**：Data Dump模式，用于指定dump算子输入还是输出数据。取值如下：

  - input：仅Dump算子输入数据
  - output：仅Dump算子输出数据，默认为output。
  - all：Dump算子输入和输出数据

    > [!NOTE]说明
    > 配置为all时，由于部分算子在执行过程中会修改输入数据，例如集合通信类算子HcomAllGather、HcomAllReduce等，因此系统在进行dump时，会在算子执行前dump算子输入，在算子执行后dump算子输出，这样，针对同一个算子，算子输入、输出的dump数据是分开落盘，会出现多个dump文件，在解析dump文件后，用户可通过文件内容判断是输入还是输出。

- **enable_dump_debug**：是否开启溢出检测功能，默认值：False。
  - True：开启溢出检测功能，从dump_path读取Dump文件保存路径，dump_path为None时会产生异常。
  - False：关闭溢出检测功能。

    > [!NOTE]说明
    >- 不能同时开启Data Dump与溢出数据采集功能，即不同时将enable_dump和enable_dump_debug参数配置为“True”。
    >- 若“enable_dump/enable_dump_debug（二选一）”参数配置为“True”，同时“enable_exception_dump”配置为了“1”（即开启普通异常算子dump）。此时，针对动态shape网络，仅“enable_exception_dump”生效；针对静态shape网络，“enable_exception_dump”与“enable_dump/enable_dump_debug（二选一）”都生效。

- **dump_debug_mode**：溢出检测模式，取值如下：
  - aicore_overflow：AI Core算子溢出检测，检测在算子输入数据正常的情况下，输出是否不正常的极大值（如float16下65500，38400，51200这些值）。一旦检测出这类问题，需要根据网络实际需求和算子逻辑来分析溢出原因并修改算子实现。
  - atomic_overflow：Atomic Add溢出检测模式，在AI Core计算完，由UB搬运到OUT时，产生的Atomic Add溢出问题。
  - all：同时进行AI Core算子溢出检测和Atomic Add溢出检测。默认值为“all”。

    > [!NOTE]说明
    > 针对Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，仅支持配置为默认值“all”。

- **precision_mode**：算子精度模式，配置要求为string类型。

  - allow_fp32_to_fp16：
    - 对于矩阵类算子：
      - 如果原图中算子精度为float32，优先降低精度到float16，如果AI Core中算子不支持float16，则继续选择float32，如果AI Core中算子不支持float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
      - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

    - 对于矢量类算子，优先保持原图精度：
      - 如果原图中算子精度为float32，则优先使用原图精度float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
      - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

  - force_fp16：

    算子同时支持float16、bfloat16和float32数据类型时，强制选择float16数据类型。**此参数仅适用于在线推理场景。**

  - force_fp32/cube_fp16in_fp32out：

    配置为force_fp32或cube_fp16in_fp32out，效果等同，该选项用来表示AI Core中该算子既支持float32又支持float16数据类型时，系统内部都会根据算子类型不同，选择不同的处理方式。cube_fp16in_fp32out为新版本中新增的，对于矩阵计算类算子，该选项语义更清晰。

    - 对于矩阵计算类算子，系统内部会按算子实现的支持情况处理：
            1. 优先选择输入数据类型为float16且输出数据类型为float32；
            2. 如果1中的场景不支持，则选择输入数据类型为float32且输出数据类型为float32；
            3. 如果2中的场景不支持，则选择输入数据类型为float16且输出数据类型为float16；
            4. 如果3中的场景不支持，则报错。

    - 对于矢量计算类算子，表示原图中算子精度为float16或bfloat16，强制选择float32。

      如果原图中存在部分算子，在AI Core中该算子的实现不支持float32，比如某算子仅支持float16类型，则该参数不生效，仍然使用支持的float16；如果在AI Core中该算子的实现不支持float32，且又配置了黑名单（precision_reduce = false），则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

  - must_keep_origin_dtype：

    保持原图精度。

    - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持float32和bfloat16，则系统内部会自动采用高精度float32。
    - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持bfloat16，则会使用float16的AI CPU算子；如果AI CPU算子也不支持，则执行报错。
    - 如果原图中某算子精度为float32，AI Core中该算子的实现不支持float32类型、仅支持float16类型，则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

  - allow_mix_precision_fp16/allow_mix_precision：

    配置为allow_mix_precision或allow_mix_precision_fp16，效果等同，均表示使用混合精度float16、bfloat16和float32数据类型来处理神经网络的过程。allow_mix_precision_fp16为新版本中新增的，语义更清晰，便于理解。

    针对原始模型中float32和bfloat16数据类型的算子，按照内置的优化策略，自动将部分float32和bfloat16的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

  - allow_mix_precision_bf16：

    表示使用混合精度bfloat16和float32数据类型来处理神经网络的过程。针对原始模型中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到bfloat16，从而在精度损失很小的情况下提升系统性能并减少内存使用；如果AI Core中算子不支持bfloat16和float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

    说明：仅Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，支持此配置。

  - allow_fp32_to_bf16：

    - 如果原图中算子精度为float32，则优先使用原图精度float32，如果AI Core中算子不支持float32，则降低精度到bfloat16；如果AI Core中算子不支持bfloat16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
    - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

      说明：Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，支持此配置。

    针对Atlas训练系列产品，默认配置项为“allow_fp32_to_fp16”。

    针对Atlas A2系列产品，默认配置项为“must_keep_origin_dtype”。

- **graph_run_mode**：图执行模式，取值：
  - 0：在线推理场景下，请配置为0。
  - 1（默认值）：训练场景下，请配置为1。

- **op_debug_level**：算子debug功能开关，取值：
  - 0：不开启算子debug功能。
  - 1：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o和.json文件），用于后续工具进行AI Core Error问题定位。

    注意：Ascend 950PR&950DT系列产品不会生成TBE指令映射文件。

  - 2：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件\*.cce、python-cce映射文件\*_loc.json、.o和.json文件），并关闭ccec编译器的编译优化开关且打开ccec调试功能（ccec编译器选项设置为-O0-g），用于后续工具进行AI Core Error问题定位。

    注意：Ascend 950PR&950DT系列产品不会生成TBE指令映射文件。

  - 3：不开启算子debug功能，且在训练脚本执行目录下的kernel_meta文件夹中保留.o和.json文件。
  - 4：不开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中**保留**.o（算子二进制文件）和.json文件（算子描述文件），生成TBE指令映射文件（算子cce文件\*.cce）和UB融合计算描述文件（\{$kernel_name\}_compute.json）。

    注意：Ascend 950PR&950DT系列产品不会生成TBE指令映射文件和UB融合计算描述文件。

    注意：
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
  - 该参数取值为0，同时设置了NPU_COLLECT_PATH环境变量的场景，执行命令当前路径**下仍旧会生成**算子编译目录kernel_meta；若设置了ASCEND_WORK_PATH环境变量，则在该环境变量指定路径下生成kernel_meta。关于环境变量的详细说明，可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。
  - debug功能开关打开场景下，若模型中含有如下通算融合算子，算子编译目录kernel_meta中，不会生成下述算子的\*.o、\*.json、\*.cce文件。

    ```text
    MatMulAllReduce
    MatMulAllReduceAddRmsNorm
    AllGatherMatMul
    MatMulReduceScatter
    AlltoAllAllGatherBatchMatMul
    BatchMatMulReduceScatterAlltoAll
    ```

- **enable_exception_dump**：是否dump异常算子数据。

  - 0：关闭异常算子数据dump功能。
  - 1：开启普通ExceptionDump，dump异常算子的输入输出数据、tensor描述信息（shape、dtype、format等）以及workspace信息。

    dump数据存储路径优先级为：环境变量NPU_COLLECT_PATH \> 环境变量ASCEND_WORK_PATH \> 默认路径（当前脚本执行路径下的extra-info目录）。

  - 2（默认值）：开启LiteExceptionDump，dump异常算子的输入输出数据、workspace信息、Tiling信息等，导出的数据用于分析AI Core Error问题。

    dump数据存储路径优先级为：环境变量ASCEND_WORK_PATH \> 默认路径（指当前脚本执行路径下的extra-info/data-dump/<device_id\>目录）。

    > [!NOTE]说明
    > 若配置了环境变量NPU_COLLECT_PATH，不论配置项“enable_exception_dump”的取值如何，都按照“1：普通ExceptionDump”进行异常算子数据dump，且dump数据存储在环境变量NPU_COLLECT_PATH的指定目录下。

- **op_select_implmode**：NPU内置算子有高精度和高性能实现方式，用户可以通过该参数配置模型编译时选择哪种算子。取值包括：
  - high_precision：表示算子选择高精度实现。高精度实现算子是指在fp16输入的情况下，通过泰勒展开/牛顿迭代等手段进一步提升算子的精度。
  - high_performance：表示算子选择高性能实现。高性能实现算子是指在fp16输入的情况下，不影响网络精度前提的最优性能实现。默认为high_performance。

- **optypelist_for_implmode**：列举算子optype的列表，该列表中的算子使用op_select_implmode参数指定的模式，当前支持的算子为Pooling、SoftmaxV2、LRN、ROIAlign，多个算子以英文逗号分隔。

    该参数需要与op_select_implmode参数配合使用，例如：

    op_select_implmode配置为high_precision。

    optypelist_for_implmode配置为Pooling。

    默认值为空，代表不使能此配置。

## 调用示例

如果在sess.run或者estimator.train之前调用get_local_rank_id/get_rank_size/get_rank_id等HCCL接口，需要先另起session执行initialize_system，进行集合通信初始化，然后在训练结束后执行shutdown_system，同时关闭session。

```python
import tensorflow as tf
from npu_bridge.npu_init import *
npu_int = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

init_sess = tf.Session(config=config)
init_sess.run(npu_int)

#调用HCCL接口...
#执行训练...

init_sess.run(npu_shutdown)
init_sess.close()
```

或者：

```python
import tensorflow as tf
from npu_bridge.npu_init import *
npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
    sess.run(npu_init)
    #调用HCCL接口...
    #执行训练...
    sess.run(npu_shutdown)
```
