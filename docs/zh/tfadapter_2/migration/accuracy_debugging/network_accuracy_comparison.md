# 整网数据比对

## 使用场景

排除以上问题后，在训练网络精度仍未达预期时，通过采集训练过程中各算子的运算结果（即Dump数据），然后和业界标准算子（如TensorFlow）运算结果进行数据偏差对比，快速定位到具体算子的精度问题。主要过程为：

![](../figures/network_data_compare.png)

## 前提条件

1. 已排除浮点异常问题，并关闭溢出检测开关。
2. 已排除融合异常问题，并恢复融合规则开关状态。
3. 已完成[工具部署](accuracy_analyzer_deployment.md)。

4. 整网数据比对前，需要先检查并去除训练脚本内部使用到的随机处理，避免由于输入数据不一致导致数据比对结果不可用。

## 基于GPU/CPU Dump标杆数据

- 在TensorFlow 2.x原始训练网络获取npy或dump数据前，要求有一套完整、可执行的标准TensorFlow模型训练工程。
- 参见[tfdbg_ascend工具的readme文档](https://gitee.com/ascend/tools/tree/master/tfdbg_ascend)安装TensorFlow 2.x的debug工具tfdbg_ascend。
- 首先要把脚本中所有的随机全部关闭，包括但不限于对数据集的shuffle，参数的随机初始化，以及某些算子的隐式随机初始化（例如dense算子），确认自己脚本内所有参数均非随机初始化。

利用TensorFlow的debug工具tfdbg_ascend生成npy文件。详细的操作方法如下：

1. 修改TensorFlow训练脚本，在调用模型部分的训练脚本.py文件中修改配置。示例代码如下。

    样例一：

    1. 导入debug插件。

        ```python
        import tfdbg_ascend as dbg
        ```

    2. 在每个step训练启动代码前配置如下代码，例如dump第5个step的数据。

        ```python
              dbg.disable()
              if current_step == 5:
                  dbg.enable()
                  dbg.set_dump_path("home/test/gpu_dump")
        ```

    样例二：

    1. 导入debug插件。

        ```python
        import tfdbg_ascend as dbg
        ```

    2. 例如dump第4个step的数据。dbg.enable不配置时，dump功能默认开启；dump路径不指定时，dump文件默认保存在训练脚本所在路径下。

        ```python
        class DumpConfig(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
            def on_batch_begin(self, batch, logs={}):
                if batch == 4:
                    dbg.enable()
                    dbg.set_dump_path("/home/test/gpu_dump")
                else:
                    dbg.disable()
        ```

    3. 注册回调函数（define callbacks ）。

        ```python
        # define callbacks
        callbacks = [
            ModelCheckpoint(
                f'models/model_epochs-{epochs}_batch-{batch_size}_loss-{loss_function}_{Mask2FaceModel.get_datetime_string()}.h5'),
            LossHistory(batch_size),
            DumpConfig()
        ]

        # fit the model
        history = self.model.fit(train_dataset, validation_data=valid_dataset, epochs=1, callbacks=callbacks, verbose=2)
        ```

2. 执行训练脚本，训练任务停止后，在指定目录下生成\*.npy文件。
3. 检查生成的npy文件命名是否符合规则，如下图所示。

    ![](../figures/query_npy_file.png "查询-npy文件")

    > [!NOTE]说明
    > - npy文件命名规则：_\{op_name\}.\{output_index\}.\{timestamp\}_.npy，其中op_name字段需满足“A-Za-z0-9_-“正则表达式规则，timestamp需满足\[0-9\]\{1,255\}正则表达式，output_index为0\~9数字组成。
    > - 如果因算子名较长，造成按命名规则生成的npy文件名超过255字符而产生文件名异常，这类算子不支持精度比对。

## 基于NPU Dump精度数据

以下操作在NPU训练环境执行，Dump数据前需要注意：

一般情况下，dump首个step的数据用作后续比对分析即可，为了避免随机权重导致比对不准确的问题，可以在训练开始前保存ckpt，并在训练时加载。如果确定是某个step的精度问题，则建议加载最靠近异常step的ckpt文件。

1. 修改“precision_tool/lib/config”目录下的config.py文件，指定需要dump数据的step。

    ```python
    # dump特定steps的数据，一般对比分析dump首层即可，即保持默认值，如需指定特定steps可以修改，例如 '0|5|10'
    TF_DUMP_STEP = '0'
    ```

    若不配置“TF_DUMP_STEP”参数，则采集所有迭代的dump数据。

2. 修改训练脚本，使能Dump数据采集。

    以下修改会同时生成Dump数据和Dump图，用于精度数据比对。

    ```python
    import precision_tool.tf_config as npu_tf_config
    npu_tf_config.npu_device_dump_config(npu_device, action='dump')
    ```

    > [!NOTE]说明
    > 除了此种方式，您也可以参考《[精度调试工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)》的方法修改训练脚本，采集Dump数据，但配置较为复杂，且采集到数据之后，需要手工提取并放在相应目录下，用于后续数据分析。注意两种方式不能重复配置。

3. 执行训练，会在precision_data/npu/debug_0目录下分别保存GE的dump图和dump数据文件。

    关于数据的后续分析，参见下文的[精度数据比对](#精度数据比对)。

## 精度数据比对

精度数据分析依赖CANN软件包中的atc工具和msaccucmp.py工具，以下操作需要在CANN开发环境执行。

1. 将precision_tool和precision_data（包括标杆数据和NPU的精度数据）文件夹上传到CANN开发环境的任意目录下，目录结构示例：

    ```text
    ├── precision_tool
    │    ├── cli.py
    │    ├── ...
    ├── precision_data
    │    ├── npu
    │    │    ├── debug_0  // 存放npu dump数据
    │    ├── tf
    │    │    ├── dump     // 存放标杆dump数据
    ```

2. 安装Python依赖。

    ```bash
    # graphviz为可选依赖，只有当需要绘制算子子图时才需要安装
    pip3 install rich graphviz
    # ubuntu/Debian
    sudo apt-get install graphviz
    # fedora/CentOS
    sudo yum install graphviz
    ```

3. 修改工具precision_tool/lib/config目录下的config.py。

    ```python
    # 依赖CANN软件包中的atc和msaccucmp.py工具，配置为CANN软件包安装目录
    # 默认CANN软件包安装在/usr/local/Ascend，可以不用修改，指定目录安装则需要修改
    CMD_ROOT_PATH = '/usr/local/Ascend'
    ```

4. 启动PrecisionTool交互命令行。

    **python3 ./precision_tool/cli.py**

    进入交互命令行界面：

    **PrecisionTool \>**

5. 执行[ac -l \[limit_num\] \(-c\)](precision_tool_ommand_ref.md#ac--l-limit_num--c)命令进行整网精度比对。

    **PrecisionTool \> ac -c**

    根据数据量大小的不同，比对过程所需时间也不同。

    对比结果会以csv的格式存放在precision_data/temp/vector_compare目录中：

    ![](../figures/precision_ac-c.png)

    您可以直接打开csv文件进行分析，具体请参考[整网精度比对结果文件说明](network_accuracy_comparison_result_file.md)。

6. 除了直接打开csv文件进行精度分析外，您也可以使用[vcs -f \[file_name\] -c \[cos_sim_threshold\] -l \[limit\]](precision_tool_ommand_ref.md#vcs--f-file_name--c-cos_sim_threshold--l-limit)命令筛选比对结果。

    vcs命令默认筛选余弦相似度小于0.98的结果，您也可以通过-c参数自定义阈值：

    ![](../figures/precision_vcs-f.png)

    - Left：表示基于NPU运行生成的dump数据的算子名。
    - Right：表示基于GPU/CPU运行生成的npy或dump数据的算子名。
    - Input和Output：表示该算子各输入输出的余弦相似度算法比对结果，范围是\[-1,1\]，比对的结果如果越接近1，表示两者的值越相近，越接近-1意味着两者的值越相反。

    从上图的比对结果可以看到，算子的输入基本一致，但第一个输出与标杆存在明显差异（余弦相似度为0.806927，小于0.98），说明该算子可能存在精度问题。

    > [!NOTE]说明
    > 当出现多个算子精度问题时，会出现N个异常算子信息，默认按照算子执行顺序排序，由于后面算子精度问题可能是因为前一个算子精度问题导致，建议用户优先分析第一个异常算子。

7. 执行[ni \(-n\) \[op_name\] -g \[graph\] -a \[attr\] -s \[save subgraph depth\]](precision_tool_ommand_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)命令，可以查询异常算子的节点信息。

    ![](../figures/precision_ni-n.png)

    ni命令可以根据传入的算子名称，得到如下关键信息：

    1. 算子类型，以上图为例，算子类型为Add。

        另外，PassName表示该算子为融合算子，对应值表示融合规则名称，OriginOp为融合前的算子，表明是由于算子融合导致精度问题。正常情况下，融合问题应该在[浮点异常检测](floating-point_exception_detection.md)阶段解决。

    2. 自动解析dump数据，打印dump数据的基础信息（max/min/mean）。
    3. 如果传入-s，则会保存一个以当前算子为中心，指定深度的子图结构，例如：

        ![](../figures/structural_drawing.png)

## 分析思路参考

整网数据比对提供了一个全网Dump数据与TF标杆数据的逐层累计比对报表，整网数据由于硬件差异本身是存在一定误差的，且误差会随着层数增多而累计，即便精度正常的网络数值上也会存在细微误差，一般采用余弦相似度做初步的可疑算子筛选（注意：余弦相似度较高也不一定说明没有问题，但较低一般代表可能存在问题），精度比对结果可以给出一个大致的分析方向。

1. 根据算子类型，可以判断该算子是否为用户自定义算子：
    - 对于自定义算子，一般由用户自行分析算子的实现逻辑是否与标杆一致，可以根据[ni \(-n\) \[op_name\] -g \[graph\] -a \[attr\] -s \[save subgraph depth\]](precision_tool_ommand_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)命令提供的算子参数信息，以及dump数据进行单算子分析。
    - 对于CANN内置算子，如果算子输入或输出类型为float16，则可以切换算子类型至float32计算。用户可以尝试以下两种方法：
      1.（推荐）方法一：通过[modify_mixlist](../../apiref/npu-global_options/accuracy_tuning.md#modify_mixlist)的修改混合精度模式算子黑白灰名单，调整算子精度模式。
      1. 方法二：通过[npu.keep_dtype_scope](../../apiref/npu-keep_dtype_scope.md)接口，指定哪些算子保持原有精度。

        ```python
        import npu_device as npu
        with npu.keep_dtype_scope():
            v = tf.add(1, 1)
        ```

2. 如果依旧无法解决，请在本源码仓提issue。
