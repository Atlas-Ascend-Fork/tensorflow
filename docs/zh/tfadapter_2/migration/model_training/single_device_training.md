# 执行单Device训练

本节介绍如何基于迁移好的TensorFlow训练脚本，在单Device上执行训练。

> [!CAUTION]注意
> 一个Device对应执行一个训练进程，当前不支持多进程在同一个Device上进行训练。

## 前提条件

- 已准备迁移好的TensorFlow训练脚本和对应数据集。
- **如果训练脚本中使用了HCCL集合通信接口**，执行训练前需要配置Device资源信息。可以通过配置文件（rank table文件）的方式或者环境变量的方式进行配置，**由于是单Device训练，所以仅配置当前一个Device资源即可**，然后启动训练进程，本章节不对此场景的执行步骤展开介绍，详细可参考[执行分布式训练](distributed_training.md)。

  **需要注意，此种场景下，若通过rank table文件方式配置资源信息，分布式环境变量“RANK_ID”固定配置为“0”，“RANK_SIZE”固定配置为“1”即可。**

## 在单Device上执行训练

1. 配置启动训练进程依赖的环境变量。

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh
    
    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
    
    export JOB_ID=10087        # 训练任务ID，用户自定义，仅支持大小写字母，数字，中划线，下划线。不建议使用以0开头的纯数字
    export ASCEND_DEVICE_ID=0  # 指定AI处理器的逻辑ID，单卡训练也可不配置，默认为0，在0卡执行训练
    ```

2. （可选）配置辅助功能环境变量。
    - 为了后续方便定位问题，启动训练脚本前用户也可以通过环境变量使能dump计算图。

        ```bash
        export DUMP_GE_GRAPH=2                  # 1：全量dump；2：不含有权重等数据的基本版dump；3：只显示节点关系的精简版dump
        export DUMP_GRAPH_PATH=/home/dumpgraph  # 通过该环境变量指定dump图文件存储路径
        ```

        训练任务启动后，会在$\{DUMP_GRAPH_PATH\}/pid_$\{pid\}_deviceid_$\{deviceid\}路径下生成若干dump图文件，包括后缀为“.pbtxt”和“.txt”的dump文件。由于dump的数据文件较多且文件都较大，若非问题定位需要，可以不生成dump图。

    - 若开发者期望程序编译运行过程中产生的文件落盘到统一存储目录，可通过环境变量ASCEND_CACHE_PATH与ASCEND_WORK_PATH分别设置共享文件的存储路径与进程独享文件的存储路径。

        ```bash
        export ASCEND_CACHE_PATH=/repo/task001/cache
        export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
        ```

        > [!NOTE]说明
        > 配置这些环境变量前，请使用**env**命令查询ASCEND_CACHE_PATH与ASCEND_WORK_PATH环境变量是否已存在，建议系统各功能使用统一的规划路径。

3. 执行训练脚本启动训练进程，例如：

    **python3 /home/xxx.py**

## 检查执行结果

1. 检查训练过程是否正常，Loss是否收敛。

    ![](../figures/single_device_loss_convergence.png)

2. 训练结束后，一般会生成如下目录和文件。
    - model目录：存放Checkpoint文件和模型文件。是否生成该目录取决于脚本实现，若训练脚本中使用saver = tf.train.Saver\(\)和saver.save\(\)保存了模型，则会生成类似如下文件。

        ![](../figures/train_result.png)

    - kernel_meta目录：用于存放算子的.o及.json文件，可用于后续问题定位。默认目录下没有这些文件，可以修改训练脚本，将运行参数op_debug_level传入3，从而保留.o和.json文件。

## 问题定位

如果运行失败，通过日志分析并定位问题。

在Host侧运行应用程序产生的运行日志路径：$HOME/ascend/log/run/plog/plog-pid_\*.log。

在Device侧运行应用程序产生的运行日志路径：$HOME/ascend/log/run/device-id/device-pid_\*.log。

$HOME为Host侧用户根目录。

一般通过ERROR级别的日志，识别问题产生模块，根据具体日志内容判定问题产生原因，错误日志样例如下所示：

![错误日志样例](../figures/error_log.png)

**表 1**  问题定位思路

| ModuleName | 出错流程 | 解决思路 |
| --- | --- | --- |
| 系统类报错 | 环境与版本配套错误 | 系统类报错，优先排查版本配套是否正确，系统是否正常安装。 |
| GE | GE图编译或校验问题 | 校验类报错，通常会给出明确的错误原因，此时需要针对性地修改网络脚本，以满足相关要求。 |
| Runtime | 环境异常导致初始化问题或图执行问题 | 对于初始化异常，优先排查当前运行环境配置是否正确，当前环境是否有其他进程占用。 |
