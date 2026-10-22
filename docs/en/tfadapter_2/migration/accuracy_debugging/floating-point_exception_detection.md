# Floating-Point Exception Detection

## Overview

At network run time, floating-point exceptions happen from time to time. That is, the loss scale decreases many times or directly to 1. In this case, analyze the overflow and underflow data to determine the problem source.

However, step-particular overflow or underflow is expected. Generally, with loss scaling enabled, the training result of that step is automatically skipped, and gradients are not updated. You can ignore such occasional errors.

The overflow/underflow detection process is shown in the following figure.

## Prerequisites

1. You have completed  [Pre-tuning Check](pre-tuning_check.md).
2. You have completed  [One-Click Accuracy Analyzer Deployment](accuracy_analyzer_deployment.md).
3. The CANN development environment installed with the CANN package has been set up.

## Overflow/Underflow Data Dump

Perform the following operations in the NPU training environment.

1. Modify the training script to enable operator overflow/underflow data collection.

    ```python
    import precision_tool.tf_config as npu_tf_config
    npu_tf_config.npu_device_dump_config(npu_device, action='overflow')
    ```

    Note: Only overflow/underflow data of AI Core operators can be collected.

2. Perform training. If overflow/underflow is detected, overflow/underflow data file will be generated in  **precision_data/overflow/dump**.

## Overflow/Underflow Data Analysis

Overflow/Underflow data analysis depends on the **atc** and **msaccucmp.py** tools in the CANN package. Perform the following operations in the CANN development environment:

1. Upload the  **precision_tool**  and  **precision_data**  folders to any directory in the CANN development environment. The following is an example of the directory structure:

    ```text
    ├── precision_tool              
    │    ├── cli.py                   
    │    ├── ...
    ├── precision_data              
    │    ├── overflow                   
    │    │    ├── dump
    ```

2. Install the Python dependencies.

    ```bash
    pip3 install rich
    ```

3. Modify **config.py** in the **precision_tool/lib/config** directory.

    ```python
    # Depend on the atc and msaccucmp.py tools in the CANN package. Generally, the tools are in the .run package installation directory. Set this directory to its parent directory.
    # By default, the CANN package is installed in /usr/local/Ascend. You can retain the path or replace the path as needed.
    CMD_ROOT_PATH = '/usr/local/Ascend'
    ```

4. Start the precision_tool command line.

    **python3 ./precision_tool/cli.py**

    Enter the command line interface:

    **PrecisionTool \>**

5. Run the following command to analyze overflow/underflow data. For details about the command, see  [precision_tool Command Reference](precision_tool_ommand_ref.md).

    **PrecisionTool \> ac**

    The analysis duration varies depending on the data size. You will see information similar to the following if operator overflow/underflow occurs.

    ![](../figures/precision_ac.png)

    In the figure:

    - Operator name: bert_encoder_layer_10_intermediate_dense_mul_FusedMulAdd
    - Operator type: FusedMulAdd
    - Overflow/Underflow status: 32, indicating floating-point overflow/underflow detected
    - Overflow/Underflow type: AI Core operator overflow/underflow as well as that of other types of operators \(such as DHA Atomic Add or L2 Atomic Add\). It is advised to resolve the AI Core operator overflow/underflow first.
    - Operator input and output information: shape, dtype, and maximum and minimum input and output values.

    > [!NOTE]NOTE
    > When overflow/underflow occurs on more than one operator, overflow/underflow information about these faulty operators is displayed in the operator execution order. It is advised to analyze the first faulty operator, as the overflow/underflow of the rest operators is usually caused by their upstream operators.

6. Run the  [pt \(-n\) \[\*.npy\]](precision_tool_ommand_ref.md#pt--n-npy)  command to check the corresponding dump block information.

    ![](../figures/precision_pt-n-npy.png)

## Analysis Principles

Before analyzing overflow/underflow data, you need to be familiar with floating-point data overflow/underflow modes of different Ascend products.

- For the Atlas training products, the default overflow/underflow mode of floating-point computation is saturation, and only the saturation mode is supported. This means when an overflow/underflow occurs during computation, the computation result is saturated to a floating-point extreme value (±MAX).
- Other series products, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.

Analyze overflow/underflow data as follows:

1. Inspect the input and output values.
    - If the input data does not contain overflow/underflow values \(saturation mode: 65504/NaN; INF/NaN mode: Inf/NaN\) but the output data does, we consider that overflow/underflow has occurred during computation.
    - If there is an overflow/underflow value among inputs, check the forward operators or the constant inputs of the user model to see whether there is an exception.
    - Otherwise, overflow/underflow may occur.

2. Analyze the exception according to the type of the faulty operator.
    - For a custom operator, analyze overflow/underflow data by yourself \(based on the operator formulas and overflow/underflow values\) to check whether the custom operator is normal.
    - For a CANN built-in operator, analyze the data as follows:
        - If the operator output data type is float16 and the dumped output values contain 65504/65500 \(saturation mode\) or Inf/NaN \(INF/NaN mode\), switch the output operator type to float32. Try either of the following methods:
            1. \(Recommended\) Method 1: Use  [modify_mixlist](../../apiref/npu-global_options/accuracy_tuning.md#modify_mixlist) to modify the blocklist, trustlist, and graylist for the operator that uses the mixed precision mode.
            2. Method 2: Use the  [npu.keep_dtype_scope](../../apiref/npu-keep_dtype_scope.md)  API to preserve the original precision for a selected operator.

                ```python
                import npu_device as npu
                with npu.keep_dtype_scope():
                    v = tf.add(1, 1)
                ```

        - If overflow/underflow does not happen in inputs and outputs, analyze the computation process according to the operator formulas. For example, AvgPool calculates the sum and mean, and overflow/underflow may occur during the sum operation, but disappear after the result is averaged.

3. If the fault persists, please raise an issue in the current source code repository.
