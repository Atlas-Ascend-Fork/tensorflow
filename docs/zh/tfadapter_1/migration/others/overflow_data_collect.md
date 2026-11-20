# 溢出数据采集

## 概述

网络模型较大的情况下，直接进行算子精度分析时，会导致Dump出的数据非常多，而且网络随机性无法完全固定，很难与标杆数据对比分析到具体哪个算子的精度问题。这种情况下，可以先开启溢出数据采集功能，溢出检测目前有三种检测模式：

- aicore_overflow：AI Core算子溢出检测模式，检测在算子输入数据正常的情况下，计算后的值是否为不正常的极大值（如float16下65500,38400,51200这些值）。一旦检测出这类问题，需要根据网络实际需求和算子逻辑来分析溢出原因并修改算子实现。
- atomic_overflow：Atomic Add溢出检测模式，在AI Core计算完，由UB搬运到OUT时，产生的Atomic Add溢出问题。
- all：同时进行AI Core算子溢出检测和Atomic Add溢出检测。

通过溢出检测结果定位到异常算子，然后再通过Data Dump功能针对性地分析该算子对应的dump数据，从而解决对应算子的精度问题。

> [!NOTE]说明
>
> - 针对Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，仅支持溢出检测模式“all”。
> - 默认训练过程中不采集溢出数据，如需采集，请参考本节内容修改训练脚本。也可以参考[浮点异常检测](../accuracy_debugging/floating-point_exception_detection.md)的方法一键式采集和分析。

## 使用注意事项

- 不能同时采集算子的dump数据和溢出数据，即不能同时开启**enable_dump**和**enable_dump_debug**。
- 开启采集溢出数据功能或者Data Dump功能都可能会产生较多结果文件，导致磁盘空间不足，请适当控制迭代次数。

## Estimator模式下采集溢出信息

Estimator模式下，通过NPURunConfig中的dump_config配置溢出检测模式，在创建NPURunConfig之前，需要实例化一个DumpConfig类，关于DumpConfig类的构造函数，请参见对应接口说明。

```python
from npu_bridge.npu_init import *

# dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限。
# enable_dump_debug：是否开启采集溢出数据的功能
# dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
dump_config = DumpConfig(enable_dump_debug = True, dump_path = "/home/test/output", dump_debug_mode = "all")
session_config=tf.ConfigProto()

config = NPURunConfig(
    dump_config=dump_config,
    session_config=session_config)
```

## sess.run模式下采集溢出信息

sess.run模式下，通过session配置项dump_path、enable_dump_debug、dump_debug_mode配置溢出检测模式：

```python
config = tf.ConfigProto()

custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True

# dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限。
custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
# enable_dump_debug：是否开启采集溢出数据的功能
custom_op.parameter_map["enable_dump_debug"].b = True
# dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF


with tf.Session(config=config) as sess:
  print(sess.run(cost))
```

## 查看与解析溢出数据

生成的溢出算子数据文件默认存储在\{dump_path\}/\{time\}/\{device_id\}/\{model_name\}/\{model_id\}/\{data_index\}目录下，例如：“/home/HwHiAiUser/output/20200808163566/0/npu_cluster_0/11/0”。如果没有采集到溢出数据，即不存在溢出情况，则不会生成上述目录。

关于溢出数据文件的介绍，以及如何解析溢出数据文件的详细操作，可参见《[精度调试工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)》中的“扩展功能 \> 溢出算子数据采集与解析”章节。
