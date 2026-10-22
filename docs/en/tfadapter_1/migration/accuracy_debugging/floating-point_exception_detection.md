# Floating-Point Exception Detection

If floating-point exceptions frequently occur during network training, you can refer to this section for detection.

## Overview

At network run time, floating-point exceptions happen from time to time. In this case, analyze the overflow and underflow data to determine the problem source.

- In dynamic loss scaling scenarios, you need to determine the problem source when the loss scaling decreases many times or directly to 1. During multi-step training, an overflow occurring in a single step is usually a normal sporadic overflow. With loss scaling enabled, the training result of such a step is automatically skipped and the gradient is not updated, so such sporadic overflows can generally be ignored.
- In static loss scaling scenarios, you must determine the source of floating-point exceptions even if there is a small amount of overflow or underflow.

  > [!NOTE]NOTE
  > For details about how to print the loss scale value, see  [Printing the Loss Scale Value](../performance_tuning/mixed_precision_training.md#printing-the-loss-scale-value).

The following figure shows the main process of overflow/underflow data detection.

**Figure  1**  Process of overflow/underflow data detection  
![](../figures/ovf_detection_flow.png)

## Prerequisites

1. You have completed  [Pre-tuning Check](pre-tuning_check.md).
2. You have completed  [Model Accuracy Analyzer Deployment](accuracy_analyzer_deployment.md).
3. The CANN development environment installed with the CANN package has been set up.

## Overflow/Underflow Data Dump

**Perform the following operations in the NPU training environment.**

1. Modify the training script to enable operator overflow/underflow data collection.

    ```bash
    # Import precision_tool/tf_config.py.
    import precision_tool.tf_config as npu_tf_config
    
    # 1. Manual network porting
    # 1.1 Estimator mode
    dump_config=npu_tf_config.estimator_dump_config(action='overflow')
    npu_config = NPURunConfig(dump_config=dump_config)
    # 1.2 Session run mode
    config = npu_tf_config.session_dump_config(config, action='overflow')
    sess = tf.Session(config)
    
    # 2. Automated network porting
    # If custom_op is not configured in the script, add the following statement in bold to the script:
    session_config = npu_tf_config.session_dump_config(session_config, action='overflow')
    # If custom_op has been configured in the script, add the following statement in bold to the script to update custom_op:
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op = npu_tf_config.update_custom_op(custom_op, action='overflow')
    
    # 2.1 Estimator mode
    run_config = tf.estimator.RunConfig(session_config=session_config,...)
    # 2.2 Session run mode
    with tf.Session(config=npu_config_proto(session_config)):
        ....
    # 2.3 tf.keras mode
    npu_keras_sess = set_keras_session_npu_config(config=session_config)
    ```

    > [!NOTE]NOTE
    >
    > - In addition to this method,  [Overflow/Underflow Data Collection](../others/overflow_data_collect.md)  provides another mode to collect overflow/underflow data. However, the configuration is complex, and you need to manually extract the overflow/underflow data and save it to the required directory for analysis. Note that the two modes are mutually exclusive.
    > - Only overflow/underflow data of AI Core operators can be collected.

2. Perform training. If overflow/underflow is detected, an overflow/underflow data file will be generated in  **precision_data/overflow/dump**.

## Overflow/Underflow Data Analysis

Overflow/Underflow data analysis depends on the ATC and  **msaccucmp.py**  tools in the CANN package.  **Perform the following operations in the CANN development environment:**

1. Upload the  **precision_tool**  and  **precision_data**  directories to any directory in the CANN development environment. The directory structure is as follows:

    ```text
    ├── precision_tool              
    │    ├── cli.py                   
    │    ├── ...
    ├── precision_data              
    │    ├── overflow                   
    │    │    ├── dump
    ```

2. Install Python third-party dependencies.

    ```bash
    pip3 install rich
    ```

3. Modify  **config.py**  in the  **precision_tool/lib/config**  directory.

    ```bash
    # Depend on the atc and msaccucmp.py tools in the CANN package. Generally, the tools are in the .run package installation directory. Set this directory to its parent directory.
    # By default, the CANN package is installed in /usr/local/Ascend. You can retain the path or replace the path as needed.
    CMD_ROOT_PATH = '/usr/local/Ascend'
    ```

4. Start the precision_tool command line.

    **python3 ./precision_tool/cli.py**

    Enter the command line interface.

    **PrecisionTool \>**

    > [!NOTE]NOTE
    > To exit, press  **Ctrl+C**.

5. Run the  [ac -l \[limit_num\] \(-c\)](precision_tool_ommand_ref.md#ac--l-limit_num--c)  command to analyze the overflow/underflow data.

    **PrecisionTool \> ac**

    The analysis duration varies depending on the data volume. You will see information similar to the following if operator overflow/underflow occurs.

    ![](../figures/op_overflow_result.png)

    In the figure:

    - Operator name: bert_encoder_layer_10_intermediate_dense_mul_FusedMulAdd
    - Operator type: FusedMulAdd
    - Overflow/Underflow status: 32, indicating floating-point overflow/underflow detected
    - Overflow/Underflow type: AI Core operator overflow/underflow as well as that of other types of operators \(such as DHA Atomic Add or L2 Atomic Add\). It is advised to resolve the AI Core operator overflow/underflow first.
    - Operator input and output information: shape, dtype, and maximum and minimum input and output values.

    > [!NOTE]NOTE
    > When overflow/underflow occurs on more than one operator, overflow/underflow information about these faulty operators is displayed in the operator execution order. It is advised to analyze the first faulty operator, as the overflow/underflow of the other operators is usually caused by their upstream operators.

6. Run the  [pt \(-n\) \[\*.npy\]](precision_tool_ommand_ref.md#pt--n-npy)  command to inspect the corresponding dump block information.

    ![](../figures/fp_exception_result.png)

## Analysis Principles

Before analyzing overflow/underflow data, you need to be familiar with floating-point data overflow/underflow modes of different Ascend products.

- For the  Atlas training products, the default overflow/underflow mode of floating-point computation is saturation, and only the saturation mode is supported. This means when an overflow/underflow occurs during computation, the computation result is saturated to a floating-point extreme value \(±MAX\).
- Other product series, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.

Analyze overflow/underflow data as follows:

1. Inspect the input and output values.
    - If the input data does not contain overflow/underflow values \(saturation mode: 65504/NaN; INF/NaN mode: Inf/NaN\) but the output data does, we consider that overflow/underflow has occurred during computation.
    - If there is an overflow/underflow value among inputs, check the forward operators or the constant inputs of the user model to see whether there is an exception. Otherwise, overflow/underflow may occur.

2. Analyze the exception according to the type of the faulty operator.
    - For a custom operator, analyze overflow/underflow data by yourself \(based on the operator formulas and overflow/underflow values\) to check whether the custom operator is normal.
    - For a CANN built-in operator, analyze the data as follows:
        - If the operator output data type is float16 and the dumped output values contain 65504/65500 \(saturation mode\) or Inf/NaN \(INF/NaN mode\), switch the output data type to float32. Try either of the following methods:
            1. \(Recommended\) Method 1: Modify the blocklist, trustlist, and graylist for the operator that uses the mixed precision mode. For details, see  [Modifying the Blocklist, Trustlist, and Graylist for Mixed Precision](../performance_tuning/mixed_precision_training.md#modifying-the-blocklist-trustlist-and-graylist-for-mixed-precision).
            2. Method 2: Use the  **keep_dtype_scope**  API to preserve the original precision for a selected operator.

                ```python
                from npu_bridge.npu_init import *
                with npu_scope.keep_dtype_scope():   
                  y = tf.mul(x1,x2)
                ```

        - If overflow/underflow does not happen in inputs and outputs, analyze the computation process according to the operator formulas. For example, AvgPool calculates the sum and mean, and overflow/underflow may occur during the sum operation, but disappear after the result is averaged.

3. If the fault persists: please raise an issue in this source code repository.
