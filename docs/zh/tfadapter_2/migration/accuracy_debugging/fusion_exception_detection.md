# 融合异常检测

## 使用场景

训练网络执行过程中，系统会根据内置的融合规则对网络中算子进行融合，以达到提高网络性能的效果。由于大多数融合是自动识别的，但可能存在未考虑到的场景，导致精度问题，因此可以尝试关闭相应融合规则，定界网络问题是否是由于融合导致。

融合异常检测的主要过程为：

![](../figures/convergence_anomaly_detection.png)

## 前提条件

1. 已完成[工具部署](accuracy_analyzer_deployment.md)。
2. 已排除浮点异常问题，并关闭溢出检测开关。

## 操作步骤

1. 修改训练脚本，关闭全部融合规则。<a id="step1"></a>

    ```python
    import precision_tool.tf_config as npu_tf_config
    npu_tf_config.npu_device_dump_config(npu_device, action='fusion_off')
    ```

2. 执行训练，检查网络精度是否有明显提高。
    - 如果网络精度有明显提高，表明是融合问题导致，接下来需要参考步骤3定位是哪个融合规则的哪一层算子融合出现了问题。
    - 如果网络精度无明显提高，需要恢复融合规则状态（将步骤1中关闭全部融合规则的代码注释掉），并进行[整网数据比对](network_accuracy_comparison.md)。

3. 定位异常融合规则。

    定位融合异常时依赖CANN软件包中的atc工具和msaccucmp.py工具，以下操作需要在CANN开发环境执行。

    1. 开启融合规则，生成dump数据和图结构文件。
        1. 恢复融合规则状态（将[步骤1](#step1)）中关闭全部融合规则的代码注释掉），参考[基于NPU Dump精度数据](network_accuracy_comparison.md#基于npu-dump精度数据)，在NPU环境执行训练，采集dump数据，该数据默认保存在precision_data/npu/debug_0目录下。
        2. 将以上数据转存到precision_data/npu/debug_1目录下。

            **mv precision_data/npu/debug_0/ precision_data/npu/debug_1**

        3. 执行atc命令，生成包含图结构信息的json文件。

            **atc --mode=5 --om=precision_data/npu/debug_1/graph/ge_proto_00005_Build.txt --json=precision_data/npu/debug_1/test_on.json**

            > [!NOTE]说明
            > 该命令行中ge_proto_00005_Build.txt文件名为举例，实际执行时，需要根据实际图文件名替换。
            > 在precision_data/npu/debug_1会存在多个类似文件名的图文件，需要找到计算图文件。一般情况选取方法为：将TensorFlow模型保存为pb文件，然后查看该模型，选取其中一个计算类算子的名字作为关键字，找包含该关键字的计算图文件；或者尝试选择文件大小最大的文件，计算图名称取计算图文件graph下的name字段值。

    2. 关闭融合规则，生成dump数据和图结构文件。
        1. 关闭全部融合规则（参考[步骤1](#step1)相关操作），再次在NPU环境执行训练，采集dump数据，该数据默认保存在precision_data/npu/debug_0目录下。

            ```python
            import precision_tool.tf_config as npu_tf_config
            npu_tf_config.npu_device_dump_config(npu_device, action='fusion_off|dump')
            ```

        2. 执行atc命令，生成包含图结构信息的json文件。

            **atc --mode=5 --om=precision_data/npu/debug_0/graph/ge_proto_00006_Build.txt --json=precision_data/npu/debug_0/test_off.json**

    3. 将融合规则关闭前后生成的dump数据进行比对。

        进入/usr/local/Ascend/cann/toolkit/tools/operator_cmp/compare目录，执行如下命令：

        **python3 msaccucmp.py compare -m precision_data/npu/debug_0/dump/20211016180613/1/ge_default_20211016180613_1/1/0 -g precision_data/npu/debug_1/dump/20211016164504/1/ge_default_20211016164504_1/1/0 -f precision_data/npu/debug_1/test_on.json -cf precision_data/npu/debug_0/test_off.json -out out_dir**

        在out_dir目录生成精度比对结果。

    4. 根据比对结果，找到精度异常的融合算子。
    5. 根据该异常算子，匹配对应计算图txt文件，找到相应融合规则名称。如果定位有困难，请在当前源码仓提issue。

4. 定位到具体融合规则后，先恢复融合规则状态（将[步骤1](#step1)中关闭全部融合规则的代码注释掉），然后仅关闭指定的融合规则。

    ```python
    import precision_tool.tf_config as npu_tf_config
    npu_tf_config.npu_device_dump_config(npu_device, action='fusion_switch')
    ```

    同时修改precision_tool/lib/config下的融合规则配置文件fusion_switch.cfg，配置样例如下，on表示开启，off表示关闭。

    ```text
    {
        "Switch":{
            "GraphFusion":{
                "ConvToFullyConnectionFusionPass":"off"
            },
            "UBFusion":{
                "TbePool2dQuantFusionPass":"off"
            }
        }
    }
    ```

    > [!NOTE]说明
    > 具体融合规则说明请参考《[图融合和UB融合规则参考](https://hiascend.com/document/redirect/CannCommunitygraphubfusionref)》。
    >
    > **注意：针对Ascend 950PR&950DT系列产品，不支持UB融合。**
