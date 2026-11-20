# Fusion Exception Detection

## Overview

At network run time, the system fuses operators according to built-in fusion patterns for better network performance. As most fusions are proceeded automatically, it is possible that your model contains an operator that is not yet covered by the fusion implementations, which impacts model accuracy. You can disable the involved fusion patterns to determine whether the problem happens in operator fusion phase.

The fusion exception detection process is shown in the following figure.

## Prerequisites

1. You have completed  [One-Click Accuracy Analyzer Deployment](accuracy_analyzer_deployment.md).
2. Floating-point exceptions have been excluded, and the overflow/underflow detection function has been disabled.

## Procedure

1. Modify the training script to disable all fusion patterns.<a id="step1"></a>

    ```python
    import precision_tool.tf_config as npu_tf_config
    npu_tf_config.npu_device_dump_config(npu_device, action='fusion_off')
    ```

2. Perform training and check whether the network accuracy improves remarkably.
    - If yes, the problem is caused by operator fusion. Locate the error fusion patterns and the corresponding operators by referring to  step3.
    - If no, restore the fusion patterns by commenting out the code for disabling all fusion patterns in step1 and go to [Network Accuracy Comparison](network_accuracy_comparison.md).

3. Locate the error fusion patterns.

    Locating fusion exceptions depends on the  **atc**  and  **msaccucmp.py**  tools in the CANN package. Perform the following operations in the CANN development environment:

    1. Enable fusion patterns to generate dump data and graph files.
        1. Restore the fusion patterns by commenting out the code for disabling all fusion patterns in [1](#step1). Start training on the NPU by referring to  [Dumping User Model on the NPU](network_accuracy_comparison.md#dumping-user-model-on-the-npu)  and collect dump data. The data will be saved to the  **precision_data/npu/debug_0**  directory by default.
        2. Copy the preceding data to the  **precision_data/npu/debug_1**  directory.

            **mv precision_data/npu/debug_0/ precision_data/npu/debug_1**

        3. Run the  **atc**  command to generate a JSON file that contains the graph structure.

            **atc --mode=5 --om=precision_data/npu/debug_1/graph/ge_proto_00005_Build.txt --json=precision_data/npu/debug_1/test_on.json**

            > [!NOTE]NOTE
            > The file name  **ge_proto_00005_Build.txt**  in the preceding command is for reference only. Replace it as required.
            > You might find multiple graph files with similar names in  **precision_data/npu/debug_1**. To select the right computational graph file, save the TensorFlow model as a  **.pb**  file and choose a random compute operator from the  **.pb**  model as the keyword. Then search for the name of the chosen operator in the generated graph files, or find the file with the largest size. The graph that gives a match is the desired computational graph file, whose name is indicated by the  **name**  field under  **graph**.

    2. Disable the fusion patterns and generate dump data and graph files.
        1. Disable all fusion patterns by referring to  [1](#step1), perform training on the NPU again, and collect dump data. The data will be saved to the  **precision_data/npu/debug_0**  directory by default.

            ```python
            import precision_tool.tf_config as npu_tf_config
            npu_tf_config.npu_device_dump_config(npu_device, action='fusion_off|dump')
            ```

        2. Run the  **atc**  command to generate a JSON file that contains the graph structure.

            **atc --mode=5 --om=precision_data/npu/debug_0/graph/ge_proto_00006_Build.txt --json=precision_data/npu/debug_0/test_off.json**

    3. Compare the dump data generated before and after fusion patterns are disabled.

        Go to the  **/usr/local/Ascend/cann/toolkit/tools/operator_cmp/compare**  directory and run the following command:

        **python3 msaccucmp.py compare -m precision_data/npu/debug_0/dump/20211016180613/1/ge_default_20211016180613_1/1/0 -g precision_data/npu/debug_1/dump/20211016164504/1/ge_default_20211016164504_1/1/0 -f precision_data/npu/debug_1/test_on.json -cf precision_data/npu/debug_0/test_off.json -out out_dir**

        Find the accuracy comparison result in the  **out_dir**  directory.

    4. Locate the fused operator that causes accuracy degradation.
    5. Based on the operator, match the corresponding computational graph file \(.txt\) and find the corresponding fusion pattern in the file. If you have any difficulty,  click  [here](https://www.hiascend.com/support)  to contact technical support.

4. After the specific fusion pattern is determined, restore all fusion patterns by commenting out the lines for disabling all fusion patterns in  [1](#step1), and then disable only the selected fusion pattern.

    ```python
    import precision_tool.tf_config as npu_tf_config
    npu_tf_config.npu_device_dump_config(npu_device, action='fusion_switch')
    ```

    Modify the fusion pattern configuration file  **fusion_switch.cfg**  in  **precision_tool/lib/config**. The following shows an example, where  **on**  indicates enabling a certain fusion pattern,  **off**  indicates disabling a certain fusion pattern.

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

    > [!NOTE]NOTE
    > For details about fusion patterns, see  [Graph Fusion and UB Fusion Patterns](https://hiascend.com/en/document/redirect/CannCommunitygraphubfusionref).
    >
    > Note: UB fusion is not supported for the  Ascend 950PR&950DT products.
