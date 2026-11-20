# Fusion Exception Detection

## Overview

At network run time, the system fuses operators according to built-in fusion patterns for better network performance. As most fusions are performed automatically, it is possible that certain scenarios are not covered, which affect accuracy. You can try disabling the corresponding fusion rules to determine whether the issue is caused by operator fusion.

The fusion exception detection process is shown in the following figure.

![](../figures/fusion_exception_detect.png)

## Prerequisites

1. You have completed  [Model Accuracy Analyzer Deployment](accuracy_analyzer_deployment.md).
2. Floating-point exceptions have been excluded, and the overflow/underflow detection function has been disabled.

## Procedure

1. Modify the training script to disable all fusion patterns.<a id="step1"></a>

    ```python
    # Import precision_tool/tf_config.py.
    import precision_tool.tf_config as npu_tf_config

    # 1. Manual network porting: disable fusion patterns.
    # 1.1 Estimator mode
    npu_config = NPURunConfig(fusion_switch_file=npu_tf_config.FUSION_OFF_FILE)
    # 1.2 Session run mode
    config = npu_tf_config.session_dump_config(config, action='fusion_off')
    sess = tf.Session(config)

    # 2. Automated network porting: disable fusion patterns.
    # If custom_op is not configured in the script, add the following statement in bold to the script:
    session_config = npu_tf_config.session_dump_config(session_config, action='fusion_off')
    # If custom_op has been configured in the script, add the following statement in bold to the script to update custom_op:
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op = npu_tf_config.update_custom_op(custom_op, action='fusion_off')

    # 2.1 Estimator mode
    run_config = tf.estimator.RunConfig(session_config=session_config,...)
    # 2.2 Session run mode
    with tf.Session(config=npu_config_proto(session_config)):
        ....
    # 2.3 tf.keras mode
    npu_keras_sess = set_keras_session_npu_config(config=session_config)
    ```

2. Perform training and check whether the network accuracy improves remarkably.
    - If yes, the problem is caused by operator fusion. Locate the error fusion patterns and the corresponding operators by referring to  [3](#step3).
    - If no, restore the fusion patterns by commenting out the code for disabling all fusion patterns in  [1](#step1)  and go to  [Network-wide Accuracy Comparison](network_accuracy_comparison.md).

3. Locate the error fusion patterns.<a id="step3"></a>

    Locating fusion exceptions depends on the ATC and  **msaccucmp.py**  tools in the CANN package. Perform the following operations in the CANN development environment:

    1. Enable fusion patterns to generate dump data and graph files.
        1. Restore the fusion patterns by commenting out the code for disabling all fusion patterns in  [1](#step1). Start training on the NPU and collect dump data by referring to  [Dumping User Model on NPU](network_accuracy_comparison.md#dumping-user-model-on-npu). The data will be saved to the  **precision_data/npu/debug_0**  directory by default.
        2. Copy the preceding data to the  **precision_data/npu/debug_1**  directory.

            **mv precision_data/npu/debug_0/ precision_data/npu/debug_1**

        3. Run the  **atc**  command to generate a JSON file that contains the graph structure.

            **atc --mode=5 --om=precision_data/npu/debug_1/graph/ge_proto_00005_Build.txt --json=precision_data/npu/debug_1/test_on.json**

            > [!NOTE]NOTE
            > The file name  **ge_proto_00005_Build.txt**  in the preceding command is for reference only. Replace it as required.
            >
            > You might find multiple graph files with similar names in  **precision_data/npu/debug_1/graph**. To select the right computational graph file, save the TensorFlow model as a .pb file and choose a random compute operator from the .pb model as the keyword. Then search for the name of the chosen operator in the generated graph files, or find the file with the largest size. The graph that gives a match is the desired computational graph file, whose name is indicated by the  **name**  field under  **graph**.

    2. Disable the fusion patterns and generate dump data and graph files.
        1. Disable all fusion patterns by referring to  [1](#step1), perform training on the NPU again, and collect dump data. The data will be saved to the  **precision_data/npu/debug_0**  directory by default.

            ```python
            # The automated porting script is used as an example.
            config = npu_tf_config.session_dump_config(config, action='fusion_off|dump')
            ```

        2. Run the  **atc**  command to generate a JSON file that contains the graph structure.

            **atc --mode=5 --om=precision_data/npu/debug_0/graph/ge_proto_00006_Build.txt --json=precision_data/npu/debug_0/test_off.json**

    3. Compare the dump data generated before and after fusion patterns are disabled.

        Go to the  **/toolkit/tools/operator_cmp/compare**  directory in the CANN software installation directory and run the following command:

        **python3 msaccucmp.py compare -m precision_data/npu/debug_0/dump/20211016180613/1/ge_default_20211016180613_1/1/0 -g precision_data/npu/debug_1/dump/20211016164504/1/ge_default_20211016164504_1/1/0 -f precision_data/npu/debug_1/test_on.json -cf precision_data/npu/debug_0/test_off.json -out out_dir**

        Find the accuracy comparison result in the  **out_dir**  directory.

    4. Locate the fused operator that causes accuracy degradation.
    5. Based on the operator, match the corresponding computational graph file \(.txt\) and find the corresponding fusion pattern in the file. If you have any difficulty:  please raise an issue in this source code repository..

4. After the specific fusion pattern is determined, restore all fusion patterns by commenting out the lines for disabling all fusion patterns in  [1](#step1), and then disable only the selected fusion pattern.

    ```python
    # Disable the selected fusion pattern.
    # Import precision_tool/tf_config.py.
    import precision_tool.tf_config as npu_tf_config

    # 1. Manual network porting
    # 1.1 Estimator mode
    npu_config = NPURunConfig(fusion_switch_file=npu_tf_config.FUSION_SWITCH_FILE)
    # 1.2 Session run mode
    config = npu_tf_config.session_dump_config(config, action='fusion_switch')
    sess = tf.Session(config)

    # 2. Automated network porting
    session_config = npu_tf_config.session_dump_config(session_config, action='fusion_switch')
    # 2.1 Estimator mode
    run_config = tf.estimator.RunConfig(session_config=session_config,...)
    # 2.2 Session run mode
    config = npu_config_proto(config_proto=session_config)
    with tf.Session(config=config) as sess:
        ....
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
    > For details about fusion patterns, see  [Graph Fusion and UB Fusion Patterns](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/graphubfusionref/atlasrr_30_0003.html).
    >
    > UB fusion is not supported for the  Ascend 950PR&950DT products.
