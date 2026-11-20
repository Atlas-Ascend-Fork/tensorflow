# 自动迁移样例

## 简介

下面介绍如何通过工具迁移ResNet50网络。

## 下载原始模型和数据集

1. 从github下载ResNet50原始模型。

    ```bash
    git clone -b r1.13.0 https://github.com/tensorflow/models.git
    ```

    假设原始模型下载到了/root/models目录下，用户可以在/root/models/official/resnet/下查看到下载到的ResNet50原始脚本：

    ![](../figures/resnet50.png)

2. 下载数据集。

    参考[https://github.com/tensorflow/models/blob/r1.13.0/official/resnet/README.md](https://github.com/tensorflow/models/blob/r1.13.0/official/resnet/README.md)的使用说明，下载数据集，具体操作为：

    ```bash
    cd /root/models/official/resnet/
    python cifar10_download_and_extract.py
    export PYTHONPATH="$PYTHONPATH:/root/models"
    ```

    数据集模型默认下载到**/tmp/cifar10_data**路径下。

## 使用迁移工具进行模型迁移

1. 在运行环境上安装工具依赖。

    ```bash
    pip3 install pandas
    pip3 install xlrd==1.2.0
    pip3 install openpyxl
    pip3 install tkintertable
    pip3 install google_pasta
    ```

2. 执行命令进行工具自动迁移。
    1. 进入迁移工具所在目录。

        ```bash
        cd  ${TFPLUGIN_INSTALL_PATH}/npu_bridge/convert_tf2npu/
        ```

        其中$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包安装路径。

    2. 进行脚本迁移。
        - 若后续执行单Device训练，执行如下命令：

            ```bash
            python3 main.py -i /root/models/official/resnet/ -o /root/models/official/ -r /root/models/official/
            ```

            - -i：被迁移的原始脚本存储路径。
            - -o：指定的迁移后脚本存储路径，该路径不能为原始脚本路径的子目录。
            - -r：指定生成的迁移报告路径。

        - 若后续需要执行分布式训练，执行如下命令：

            ```bash
            python3 main.py -i /root/models/official/resnet/ -o /root/models/official/ -r /root/models/official/ -d tf_strategy
            ```

            其中“-d”代表原始脚本使用的分布式策略，“tf_strategy”表示原始脚本使用的是tf.distribute.Strategy分布式策略。

3. 在/root/models/official/report_npu_\*\*\*下查看迁移报告。

4. 将/root/models/official/resnet目录下的原始脚本替换为迁移后脚本。

    迁移后脚本存放在/root/models/official/resnet_npu_\*\*\*目录下。

    执行训练前，需要将resnet_npu_xxx目录脚本拷贝替换到resnet目录下，从而保证脚本能正常执行，例如：

    1. 备份resnet目录下原始脚本。
    2. 将迁移后脚本替换原始脚本，例如：

        cp /root/models/official/resnet_npu_\*\*\*/\*.py /root/models/official/resnet

## 执行单Device训练

1. 由于原始脚本支持分布式训练，迁移后的脚本中存在HCCL集合通信接口，所以执行训练前需要配置Device资源信息。

    本示例以通过rank table文件配置资源信息为例，rank table中需包含一个Device资源，文件名举例：rank_table_1p.json，配置文件举例：

    ```json
    {
    "server_count":"1",
    "server_list":
    [
       {
            "device":[
                           {
                            "device_id":"0",
                            "device_ip":"192.168.1.8",
                            "rank_id":"0"
                            }
                      ],
             "server_id":"10.0.0.10"
        }
    ],
    "status":"completed",
    "version":"1.0"
    }
    ```

    rank table配置文件的详细介绍请参考《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》的“相关参考 \> 集群信息配置”章节。

2. 配置训练进程启动依赖的环境变量。

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # 当前脚本所在路径，例如：
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export JOB_ID=10087        # 训练任务ID，用户自定义，仅支持大小写字母，数字，中划线，下划线。不建议使用以0开头的纯数字
    export ASCEND_DEVICE_ID=0  # 指定AI处理器的逻辑ID，单卡训练也可不配置，默认为0，在0卡执行训练
    export RANK_ID=0           # 指定训练进程在集合通信进程组中对应的rank标识序号，单卡训练固定配置为0
    export RANK_SIZE=1         # 指定当前训练进程对应的Device在本集群大小，单卡训练固定配置为1
    export RANK_TABLE_FILE=/root/rank_table_1p.json # 如果用户原始训练脚本中使用了hvd接口或tf.data.Dataset对象的shard接口，需要配置，否则无需配置，需要注意rank_table中的参数“device_id”优先级高于环境变量“ASCEND_DEVICE_ID”。
    ```

3. 执行训练脚本启动训练进程：

    **python3 /root/models/official/resnet**/**cifar10_main.py**

4. 检查训练过程是否正常，Loss是否收敛。

    ![](../figures/sample_result1.png)

5. 训练结束后，在**tmp/cifar10_model**下生成Checkpoint文件。

    ![](../figures/sample_result2.png)

## 在两个Device上执行分布式训练

1. 准备包含两个Device的rank table资源信息配置文件，文件名举例：rank_table_2p.json，配置文件举例：

    ```json
    {
    "server_count":"1",
    "server_list":
    [
       {
            "device":[
                           {
                            "device_id":"0",     // 此配置的优先级高于环境变量“ASCEND_DEVICE_ID”
                            "device_ip":"192.168.1.8",
                            "rank_id":"0"
                            },
                            {
                             "device_id":"1",
                             "device_ip":"192.168.1.9",   // 两个Device需要处于同一网段，0卡和1卡为同一网段
                             "rank_id":"1"
                             }
                      ],
             "server_id":"10.0.0.10"
        }
    ],
    "status":"completed",
    "version":"1.0"
    }
    ```

    rank table配置文件的详细介绍请参考《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》的“相关参考 \> 集群信息配置”章节。

2. 在不同的shell窗口依次启动不同的训练进程。

    启动训练进程0：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # 当前脚本所在路径，例如：
    export PYTHONPATH=/root/models:$PYTHONPATH
    export JOB_ID=10087

    export RANK_ID=0
    export RANK_SIZE=2
    export RANK_TABLE_FILE=/root/rank_table_2p.json
    python3 /root/models/official/resnet/cifar10_main.py
    ```

    启动训练进程1：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # 当前脚本所在路径，例如：
    export PYTHONPATH=/root/models:$PYTHONPATH
    export JOB_ID=10087

    export RANK_ID=1
    export RANK_SIZE=2
    export RANK_TABLE_FILE=/root/rank_table_2p.json
    python3 /root/models/official/resnet/cifar10_main.py
    ```

3. 训练结束后，每个Device的日志信息如下所示，代表训练结束，可查看Loss是否收敛。

    ![](../figures/sample_result3.png)

    最终会在**tmp/cifar10_model**下生成Checkpoint文件。
