# Accuracy Comparison

## dump_config.enable_dump

Whether to enable the data dump function.

- True: enabled. The dump file path is read from dump_path.
- False (default): disabled.
  > [!NOTE]NOTE
  >
  > - Data dump and overflow/underflow data collection cannot be enabled at the same time.
  > - That is, dump_config.enable_dump and dump_config.enable_dump_debug cannot be set to True at the same time.If either dump_config.enable_dump or dump_config.enable_dump_debug is set to True and enable_exception_dump is set to 1 (indicating that common ExceptionDump function is enabled): For dynamic-shape networks, only enable_exception_dump takes effect. For static-shape networks, enable_exception_dump and either of dump_config.enable_dump and dump_config.enable_dump_debug take effect.

Example:

```python
npu.global_options().dump_config.enable_dump=True
```

## dump_config.dump_mode

Data dump mode, used to specify whether to dump the input or output data of the operator.

The values are as follows:

- input: dumps only operator inputs.
- output (default): dumps only operator outputs.
- all: dumps both operator inputs and outputs.

> [!NOTE]NOTE
> If this parameter is set to all, the input data of some operators, such as collective communication operators HcomAllGather and HcomAllReduce, will be modified during execution. Therefore, the system dumps the operator input before operator execution and dumps the operator output after operator execution. In this way, the dumped input and output data of the same operator is flushed to disks separately, and multiple dump files are generated. After parsing the dump files, you can determine whether the data is an input or output based on the file content.

Example:

```python
npu.global_options().dump_config.dump_mode="all"
```

## dump_config.dump_path

Dump path. Required when enable_dump or enable_dump_debug is set to True.

Create the specified path in advance in the environment (either container or host) where training is performed. The running user configured during installation must have the read and write permissions on this path. The path can be an absolute path or a path relative to the path where the training script is executed.

- An absolute path starts with a slash (/), for example, /home/HwHiAiUser/output.
- A relative path starts with a directory name, for example, output.

Example:

```python
npu.global_options().dump_config.dump_path = "/home/HwHiAiUser/output"
```

## dump_config.dump_step

Specify which iterations' Data Dump data to collect.

Separate multiple iterations with "|", for example: 0|5|10; you can also use "-" to specify a range of iterations, for example: 0|3-5|10.

If this parameter is not configured, dump data from all iterations will be collected.

Example:

```python
npu.global_options().dump_config.dump_step="0|5"
```

## dump_config.dump_data

Type of operator content to dump.

- tensor (default): dumps operator data.
- stats: dumps operator statistics. The result file is in .csv format.

In large-scale training scenarios, dumping a large amount of data takes a long time. You can dump the statistics of all operators, identify the operators that may be abnormal based on the statistics, and then dump the input or output data of these abnormal operators.

Example:

```python
npu.global_options().dump_config.dump_data = "stats"
```

## dump_config.dump_layer

Name of the operator to dump. Multiple operator names are separated by spaces. If this option is not set, all operators are dumped by default.

If the input of the specified operator involves the data operator, the data operator information is also dumped.

Example:

```python
npu.global_options().dump_config.dump_layer = "nodename1 nodename2 nodename3"
```

## dump_config.enable_dump_debug

Overflow/underflow data collection enable.

- True: enabled. The dump file path is read from dump_path. An abnormality occurs if dump_path is None.
- False (default): disabled.

> [!NOTE]NOTE
>
> - Data dump and overflow/underflow data collection cannot be enabled at the same time. That is, dump_config.enable_dump and dump_config.enable_dump_debug cannot be set to True at the same time.
> - If either dump_config.enable_dump or dump_config.enable_dump_debug is set to True and enable_exception_dump is set to 1 (indicating that common ExceptionDump function is enabled): For dynamic-shape networks, only enable_exception_dump takes effect. For static-shape networks, enable_exception_dump and either of dump_config.enable_dump and dump_config.enable_dump_debug take effect.

Example:

```python
npu.global_options().dump_config.enable_dump_debug=True
```

## dump_config.dump_debug_mode

Overflow/Underflow detection mode. The values are as follows:

- aicore_overflow: detects AI Core operator overflow/underflow, that is, detecting whether abnormal extreme values (such as 65500, 38400, and 51200 in float16) are output with normal inputs. Once such a fault is detected, analyze the cause of the overflow/underflow and modify the operator implementation based on the network requirements and operator logic.
- atomic_overflow: detects Atomic Add overflow/underflow. Atomic Add overflow/underflow is detected when data is transferred from the UB to OUT after AI Core computation.
- all: detects overflow/underflow of both AI Core operators and Atomic Add. The default value is all.

  > [!NOTE]NOTE
  > For the Ascend 950PR&950DT products, Atlas A3 products, and Atlas A2 products, only the default value all can be used.

Example:

```python
npu.global_options().dump_config.dump_debug_mode="aicore_overflow"
```

## fusion_switch_file

Directory of the fusion switch configuration file, including the file name.

The value can contain letters, digits, underscores (_), hyphens (-), and periods (.).

The built-in graph fusion and UB fusion patterns are enabled by default. You can disable selected fusion patterns in the configuration file as needed. For details about fusion patterns that can be disabled, see [Graph Fusion and UB Fusion Patterns](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/graphubfusionref/atlasrr_30_0003.html).

Note: UB fusion is not supported for the Ascend 950PR&950DT products.

Example:

 ```python
 npu.global_options().fusion_switch_file="/home/test/fusion_switch.cfg"
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
2. To disable all fusion patterns except selected ones, refer to the following example. (That is, the priority of a single fusion pattern configured in the configuration file is higher than that of "ALL".){

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

If the TensorFlow network is quantized by the AMCT tool, this parameter can be used to specify whether to collect the dump data before quantization.

- 0 (default): The input and output before quantization may be optimized during graph compilation. In this case, the dump data before quantization cannot be obtained.
- 1: After this function is enabled, the dump data before quantization can be collected.

Example:

```python
npu.global_options().quant_dumpable="1"
```

> [!NOTE]NOTE
> This parameter applies only to online inference scenarios.
> When data dump is enabled, you can set this parameter to 1 to ensure that the dump data before quantization can be collected.
