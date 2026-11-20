# Random Error Detection

## Overview

At network run time, the calculation with the same inputs may produce different outputs.

If such random issues happen, run training twice and collect operator results \(dump data\) from both runs. Compare the results to quickly identify operators that cause randomness.

## Procedure

1. Run training on the NPU and collect dump data by referring to  [Dumping User Model on the NPU](network_accuracy_comparison.md#dumping-user-model-on-the-npu). The data will be saved to the  **precision_data/npu/debug_0**  directory by default.
2. Copy the preceding data to the  **precision_data/npu/debug_1**  directory.

    **mv precision_data/npu/debug_0/ precision_data/npu/debug_1**

3. Run training on the NPU again and collect dump data. The data will be saved to the  **precision_data/npu/debug_0**  directory by default.
4. Start the precision_tool command line.

    **python3 ./precision_tool/cli.py**

    Enter the command line interface:

    **PrecisionTool \>**

    > [!NOTE]NOTE
    > To exit, press  **Ctrl+C**.

5. Run the  [vc -lt \[left_path\] -rt \[right_path\] -g \[graph\]](precision_tool_command_ref.md#vc--lt-left_path--rt-right_path--g-graph)  command to compare the data of the entire network.

    **vc -lt precision_data/npu/debug_1/dump/20211016164504/1/ge_default_20211016164504_1/1/0 -rt precision_data/npu/debug_0/dump/20211016180613/1/ge_default_20211016180613_1/1/0**

    The accuracy comparison result is generated in the  **out_dir**  directory. For details about how to analyze data, see  [Network Accuracy Comparison Result File](network_accuracy_comparison_result_file.md).

6. For the preceding results, alternatively use the  [ni \(-n\) \[op_name\] -s \[save sub graph deep\]](precision_tool_command_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)  command of precision_tool for per-layer analysis.

    **python3 precision_tool/cli.py**

    **PrecisionTool \>  ni xxx**

    If both  **debug_0**  and  **debug_1**  exist in the  **precision_data/npu/**  directory, the  **ni**  command parses the dump files, which are named after the same operator, in the two directories. The parsing result shows clearly the differences.

    ![](../figures/precision_ni_n.png)

## Analysis Principles

The cosine similarity from network-wide comparison results is generally used to preliminarily screen suspicious operators. A high cosine similarity does not rule out issues, whereas a low cosine similarity usually indicates potential problems. The results provide a general direction for analysis.

1. Determine whether an error operator is a custom operator based on the operator type.
    - For a custom operator, check that the implementation logic of the operator is consistent with that of the benchmark by inspecting the command output of  [ni \(-n\) \[op_name\] -s \[save sub graph deep\]](precision_tool_command_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)  or the dump analysis report.
    - For a built-in CANN operator, if the operator input or output type is float16, you can switch the operator type to float32 by using either of the following methods:
        1. \(Recommended\) Method 1: Use  [modify_mixlist](../../apiref/npu-global_options/accuracy_tuning.md#modify_mixlist) to modify the blocklist, trustlist, and graylist for the operator that uses the mixed precision mode.
        2. Method 2: Use the  [npu.keep_dtype_scope](../../apiref/npu-keep_dtype_scope.md)  API to preserve the original precision for a selected operator.

            ```python
            import npu_device as npu
            with npu.keep_dtype_scope():
                v = tf.add(1, 1)
            ```

2. If the fault persists,  please raise an issue in this source code repository.
