# 随机错误检测

## 使用场景

网络执行过程中，可能存在部分计算过程在相同输入的情况下给出了不同的输出的问题。

当出现以上随机问题时，可以通过执行两次训练，并分别采集各算子的运算结果（即dump数据）。通过比对分析，从而快速定位到导致随机问题的算子层。

## 操作步骤

1. 参考[基于NPU Dump精度数据](network_accuracy_comparison.md#基于npu-dump精度数据)，在NPU环境执行训练，采集dump数据，该数据默认保存在precision_data/npu/debug_0目录下。
2. 将以上数据转存到precision_data/npu/debug_1目录下。

    **mv precision_data/npu/debug_0/ precision_data/npu/debug_1**

3. 再次在NPU环境执行训练，采集dump数据，该数据默认保存在precision_data/npu/debug_0目录下。
4. 启动PrecisionTool交互命令行。

    **python3 ./precision_tool/cli.py**

    进入交互命令行界面：

    **PrecisionTool \>**

    > [!NOTE]说明
    > 如需退出，可按下Ctrl+C组合键。

5. 使用[vc -lt \[left_path\] -rt \[right_path\] -g \[graph\]](precision_tool_command_ref.md#vc--lt-left_path--rt-right_path--g-graph)命令进行整网数据对比。

    **vc -lt precision_data/npu/debug_1/dump/20211016164504/1/ge_default_20211016164504_1/1/0 -rt precision_data/npu/debug_0/dump/20211016180613/1/ge_default_20211016180613_1/1/0**

    在out_dir目录生成精度比对结果，可参考[整网精度比对结果文件说明](network_accuracy_comparison_result_file.md)进行数据分析。

6. 针对以上结果，还可以使用precision_tool的[ni \(-n\) \[op_name\] -g \[graph\] -a \[attr\] -s \[save subgraph depth\]](precision_tool_command_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)命令进行单层数据比对分析。

    **python3 precision_tool/cli.py**

    **PrecisionTool \>  ni xxx**

    当precision_data/npu/目录下同时存在debug_0和debug_1的时候，ni命令会同时解析两个文件夹下相同算子名的dump文件，从该解析结果中，可以清晰地看出数据差异。

    ![](../figures/precision_ni_n.png)

## 分析思路参考

基于整网比对结果，一般采用余弦相似度做初步的可疑算子筛选（注意：余弦相似度较高也不一定说明没有问题，但较低一般代表可能存在问题），精度比对结果可以给出一个大致的分析方向。

1. 根据算子类型，可以判断该算子是否为用户自定义算子：
    - 对于自定义算子，一般由用户自行分析算子的实现逻辑是否与标杆一致，可以根据[ni \(-n\) \[op_name\] -s \[save sub graph deep\]](precision_tool_command_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)命令提供的算子参数信息，以及dump数据进行单算子分析。
    - 对于CANN内置算子，如果算子输入或输出类型为float16，则可以切换算子类型至float32计算，用户可以尝试以下两种方法：
        1. （推荐）方法一：通过[modify_mixlist](../../apiref/npu-global_options/accuracy_tuning.md#modify_mixlist)修改混合精度模式算子黑白灰名单，调整算子精度模式。
        2. 方法二：通过[npu.keep_dtype_scope](../../apiref/npu-keep_dtype_scope.md)接口，指定哪些算子保持原有精度。

            ```python
            import npu_device as npu
            with npu.keep_dtype_scope():
                v = tf.add(1, 1)
            ```

2. 如果依旧无法解决，请在当前开源仓提issue。
