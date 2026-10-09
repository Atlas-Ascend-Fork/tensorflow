# 精度比对

## dump_config.enable_dump

是否开启Data Dump功能。

- True：开启Data Dump功能，从dump_path读取Dump文件保存路径。
- False（默认值）：关闭Data Dump功能。
    > [!NOTE]说明
    >
    > - 不能同时开启Data Dump与溢出数据采集功能，即不同时将dump_config.enable_dump和dump_config.enable_dump_debug参数配置为“True”。
    > - 若“dump_config.enable_dump/dump_config.enable_dump_debug（二选一）”参数配置为“True”，同时“enable_exception_dump”配置为了“1”（即开启普通ExceptionDump）。此时，针对动态shape网络，仅“enable_exception_dump”生效；针对静态shape网络，“enable_exception_dump”与“dump_config.enable_dump/dump_config.enable_dump_debug（二选一）”都生效。

配置示例：

```python
npu.global_options().dump_config.enable_dump=True
```

## dump_config.dump_mode

Data Dump模式，用于指定dump算子输入还是输出数据，默认为output。

取值如下：

- input：仅dump算子输入数据
- output：仅dump算子输出数据
- all：dump算子输入和输出数据

> [!NOTE]说明
> 配置为all时，由于部分算子在执行过程中会修改输入数据，例如集合通信类算子comAllGather、HcomAllReduce等，因此系统在进行dump时，会在算子执行前dump子输入，在算子执行后dump算子输出，这样，针对同一个算子，算子输入、输出的ump数据是分开落盘，会出现多个dump文件，在解析dump文件后，用户可通过文件内判断是输入还是输出。

配置示例：

```python
npu.global_options().dump_config.dump_mode="all"
```

## dump_config.dump_path

Dump文件保存路径。enable_dump或enable_dump_debug为True时，该参数必须配置。

该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装配置的运行用户具有读写权限，支持配置绝对路径或相对路径（相对执行命令行的当前路径）。

- 绝对路径配置以“/”开头，例如：/home/HwHiAiUser/output。
- 相对路径配置直接以目录名开始，例如：output。

配置示例：

```python
npu.global_options().dump_config.dump_path = "/home/HwHiAiUser/output"
```

## dump_config.dump_step

指定采集哪些迭代的Data Dump数据。

多个迭代用“|”分割，例如：0|5|10；也可以用"-"指定迭代范围，例如：0|3-5|0。

若不配置该参数，表示采集所有迭代的dump数据。

配置示例：

```python
npu.global_options().dump_config.dump_step="0|5"
```

## dump_config.dump_data

指定算子dump内容类型，取值：

- tensor: dump算子数据，默认为tensor。
- stats: dump算子统计数据，结果文件为csv格式。

大规模训练场景下，通常dump数据量太大并且耗时长，可以先dump所有算子的统数据，根据统计数据识别可能异常的算子，然后再指定dump异常算子的inputoutput数据。

配置示例：

```python
npu.global_options().dump_config.dump_data = "stats"
```

## dump_config.dump_layer

指定需要dump的算子。取值为算子名，多个算子名之间使用空格分隔。若不配置此字段，默认dump全部算子。

若指定的算子其输入涉及data算子，会同时将data算子信息dump出来。

配置示例：

```python
npu.global_options().dump_config.dump_layer = "nodename1 nodename2 nodename3"
```

## dump_config.enable_dump_debug

溢出检测场景下，是否开启溢出数据采集功能。

- True：开启采集溢出数据的功能，从dump_path读取Dump文件保存路径dump_path为None时会产生异常。
- False（默认值）：关闭采集溢出数据的功能。

> [!NOTE]说明
>
>- 不能同时开启Data Dump与溢出数据采集功能，即不同时将dump_configenable_dump和dump_config.enable_dump_debug参数配置为“True”。
>- 若“dump_config.enable_dump/dump_config.enable_dump_debug（二一）”参数配置为“True”，同时“enable_exception_dump”配置为了“1”（即开普通ExceptionDump）。此时，针对动态shape网络，“enable_exception_dump”生效；针对静态shape网络“enable_exception_dump”与“dump_config.enable_dump/dump_configenable_dump_debug（二选一）”都生效。

配置示例：

```python
npu.global_options().dump_config.enable_dump_debug=True
```

## dump_config.dump_debug_mode

溢出检测模式，取值如下：

- aicore_overflow：AI Core算子溢出检测，检测在算子输入数据正常的情下，输出是否不正常的极大值（如float16下65500,38400,51200这些值）。一检测出这类问题，需要根据网络实际需求和算子逻辑来分析溢出原因并修改算子现。
- atomic_overflow：Atomic Add溢出检测模式，在AI Core计算完，由UB运到OUT时，产生的Atomic Add溢出问题。
- all：同时进行AI Core算子溢出检测和Atomic Add溢出检测。默认值“all”。

  > [!NOTE]说明
  > 针对Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，仅支持配置为默认值“all”。

配置示例：

```python
npu.global_options().dump_config.dump_debug_mode="aicore_overflow"
```

## fusion_switch_file

融合开关配置文件路径以及文件名。

格式要求：支持大小写字母（a-z，A-Z）、数字（0-9）、下划线（_）、中划线（-）、句点（.）、中文字符。

系统内置了一些图融合和UB融合规则，均为默认开启，可以根据需要关闭指定的融合规则，当前可以关闭的融合规则请参见《[图融合和UB融合规则参考](https://hiascend.com/document/redirect/CannCommunitygraphubfusionref)》。

**注意：Ascend 950PR&950DT系列产品不支持UB融合。**

配置示例：

 ```python
 npu.global_options().fusion_switch_file="/home/test/fusion_switch.cfg"
 ```

配置文件样例fusion_switch.cfg如下所示_，_on表示开启，off表示关闭。

```text
{
    "Switch":{
        "GraphFusion":{
            "RequantFusionPass":"on",
            "ConvToFullyConnectionFusionPass":"off",
            "SoftmaxFusionPass":"on",
            "NotRequantFusionPass":"on",
            "ConvConcatFusionPass":"on",
            "MatMulBiasAddFusionPass":"on",
            "PoolingFusionPass":"on",
            "ZConcatv2dFusionPass":"on",
            "ZConcatExt2FusionPass":"on",
            "TfMergeSubFusionPass":"on"
        },
        "UBFusion":{
            "TbePool2dQuantFusionPass":"on"
        }
    }
}
```

同时支持用户一键关闭融合规则：

```text
{
    "Switch":{
        "GraphFusion":{
            "ALL":"off"
        },
        "UBFusion":{
            "ALL":"off"
         }
    }
}
```

需要注意的是：

1. 关闭某些融合规则可能会导致功能问题，因此此处的一键式关闭仅关闭系统部分融规则，而不是全部融合规则。
2. 一键式关闭融合规则时，可以同时开启部分融合规则（即配置文件中针对单个融合则配置的优先级高于“ALL”）：

   ```text
   {
       "Switch":{
           "GraphFusion":{
               "ALL":"off",
               "SoftmaxFusionPass":"on"
           },
           "UBFusion":{
               "ALL":"off",
               "TbePool2dQuantFusionPass":"on"
           }
       }
   }
   ```

## quant_dumpable

如果TensorFlow网络是经过AMCT工具量化后的网络，可通过此参数控制是否采集量化前的dump数据。

- 0（默认值）：图编译过程中可能优化量化前的输入输出，此时无法获取量化前dump数据。
- 1：开启此配置后，可确保能够采集量化前的dump数据。
  
配置示例：

```python
npu.global_options().quant_dumpable="1"
```

> [!NOTE]说明
> 此参数仅适用于在线推理场景下使用。
> 开启Data Dump的场景下，可通过将此配置项配置为“1”，确保可以采集量化前的ump数据。
