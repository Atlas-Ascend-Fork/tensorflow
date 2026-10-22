# 精度比对

## dump_config

dump开关，用户在创建NPURunConfig之前，可以实例化一个DumpConfig类进行dump的配置。DumpConfig类的构造函数，请参见[DumpConfig构造函数](../dumpconfig_constructor.md)。

配置示例：

```python
config = NPURunConfig(dump_config=dump_config)
```

## quant_dumpable

如果TensorFlow网络是经过AMCT工具量化后的网络，可通过此参数控制是否采集量化前的dump数据。

- 0（默认值）：图编译过程中可能优化量化前的输入输出，此时无法获取量化前的dump数据。
- 1：开启此配置后，可确保能够采集量化前的dump数据。

配置示例：

```python
config = NPURunConfig(quant_dumpable="1")
```

> [!NOTE]说明
>
> - 此参数仅适用于在线推理场景下使用。
> - 开启Data Dump的场景下，可通过将此配置项配置为“1”，确保可以采集量化前的dump数据。

## fusion_switch_file

融合开关配置文件路径以及文件名。

格式要求：支持大小写字母（a-z，A-Z）、数字（0-9）、下划线（_）、中划线（-）、句点（.）、中文字符。

系统内置了一些图融合和UB融合规则，均为默认开启，可以根据需要关闭指定的融合规则，当前可以关闭的融合规则请参见《[图融合和UB融合规则参考](https://hiascend.com/document/redirect/CannCommunitygraphubfusionref)》。

**注意：针对Ascend 950PR&950DT系列产品，不支持UB融合。**

配置示例：

```python
config = NPURunConfig(fusion_switch_file="/home/test/fusion_switch.cfg")
```

配置文件样例fusion_switch.cfg如下所示，on表示开启，off表示关闭。

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

1. 关闭某些融合规则可能会导致功能问题，因此此处的一键式关闭仅关闭系统部分融合规则，而不是全部融合规则。
2. 一键式关闭融合规则时，可以同时开启部分融合规则（即配置文件中针对单个融合规则配置的优先级高于“ALL”）：

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

## buffer_optimize

高级开关，是否开启buffer优化，仅适用于在线推理场景。

- l2_optimize：表示开启buffer优化，默认为l2_optimize。
- off_optimize：表示关闭buffer优化。

配置示例：

```python
config = NPURunConfig(buffer_optimize="l2_optimize")
```
