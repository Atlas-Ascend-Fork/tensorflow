# Accuracy Comparison

## dump_config

Dump configuration. Before creating NPURunConfig, you can instantiate a DumpConfig class for dump configuration. For details about the constructor of the DumpConfig class, see [DumpConfig Constructor](../dumpconfig_constructor.md).

Example:

```python
config = NPURunConfig(dump_config=dump_config)
```

## quant_dumpable

If the TensorFlow network is quantized by the AMCT tool, this parameter can be used to specify whether to collect the dump data before quantization.

- 0 (default): The input and output before quantization may be optimized during graph compilation. In this case, the dump data before quantization cannot be obtained.
- 1: After this function is enabled, the dump data before quantization can be collected.

Example:

```python
config = NPURunConfig(quant_dumpable="1")
```

> [!NOTE]NOTE
>
> - This parameter applies only to online inference scenarios.
> - When data dump is enabled, you can set this parameter to 1 to ensure that the dump data before quantization can be collected.

## fusion_switch_file

Directory of the fusion switch configuration file, including the file name.

The value can contain letters, digits, underscores (_), hyphens (-), and periods (.).

The built-in graph fusion and UB fusion patterns are enabled by default. You can disable selected fusion patterns in the configuration file as needed. For details about fusion patterns that can be disabled, see [Graph Fusion and UB Fusion Patterns](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/graphubfusionref/atlasrr_30_0003.html).

Note: UB fusion is not supported for the Ascend 950PR&950DT products.

Example:

```python
config = NPURunConfig(fusion_switch_file="/home/test/fusion_switch.cfg")
```

The following is a template of the fusion_switch.cfg configuration file. on indicates that a fusion pattern is enabled, and off indicates that a fusion pattern is disabled.

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

To disable all fusion patterns at a time, refer to this configuration file example.

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

Notes:

1. Some built-in fusion patterns are not switchable due to functionality restrictions and these fusion patterns will remain enabled despite user's switch settings.
2. To disable all fusion patterns except selected ones, refer to the following example. (That is, the priority of a single fusion pattern configured in the configuration file is higher than that of "ALL".)

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

Whether to enable buffer optimization. This is an advanced switch and applies only to online inference scenarios.

- l2_optimize (default): enabled
- off_optimize: disabled.

Example:

```python
config = NPURunConfig(buffer_optimize="l2_optimize")
```
