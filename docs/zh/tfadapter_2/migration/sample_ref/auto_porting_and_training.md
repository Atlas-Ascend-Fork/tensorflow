# 自动迁移与训练

## 简介

下面介绍如何通过工具迁移ResNet50网络。

## 下载原始模型和数据集

1. 从github下载ResNet50原始模型。

    ```bash
    git clone -b r2.6.0 https://github.com/tensorflow/models.git
    ```

    假设原始代码下载到了/root/models目录下，用户可以在/root/models/official/vision/image_classification/resnet/下查看到下载好的脚本：

    ![](../figures/resnet50_download.png)

2. 下载数据集。

    参考[ImageNet数据集](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy)的使用说明，下载ImageNet2012数据集并使用imagenet_to_gcs.py脚本转换为TFRecord。

    将处理好的数据集放到/root/models/data/imagenet_TF/路径下。

## 使用迁移工具进行模型迁移

1. 阅读[使用限制](../script_migration/automated_porting.md#了解自动迁移工具)，需要在工具迁移前手工添加数据集分片操作：

    ```python
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    import npu_device as npu
    # NPU添加的shard逻辑，会根据集群数量，对数据集和全局batch进行切分
    dataset, batch_size = npu.distribute.shard_and_rebatch_dataset(dataset, batch_size)
    #if input_context:
    #  logging.info(
    #      'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
    #      input_context.input_pipeline_id, input_context.num_input_pipelines)
    #  dataset = dataset.shard(input_context.num_input_pipelines,
    #                          input_context.input_pipeline_id)
    ```

2. 在运行环境上安装工具依赖。

    ```bash
    pip3 install pandas
    pip3 install openpyxl
    pip3 install google_pasta
    ```

3. 执行命令进行工具自动迁移。
    1. 进行迁移工具所在目录。

       ```bash
       cd ${TFPLUGIN_INSTALL_PATH}/npu_device/convert_tf2npu/
       ```

       其中\$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包安装路径。

    2. 进行脚本迁移。
        - 若后续执行单Device训练，执行如下命令：

            ```bash
            python3 main.py -i /root/models/official/vision/image_classification/resnet/ -o /root/models/resnet50/ -r /root/models/resnet50/ -m /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
            ```

        - 若后续需要执行分布式训练，执行如下命令：

            ```bash
            python3 main.py -i /root/models/official/vision/image_classification/resnet/ -o /root/models/resnet50/ -r /root/models/resnet50/ -m /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py -d tf_strategy
            ```

            其中“-d”代表原始脚本使用的分布式策略，“tf_strategy”表示原始脚本使用的是tf.distribute.Strategy分布式策略。

4. 在`/root/models/resnet50/report_npu_***`下查看迁移报告。

5. 在`/root/models/resnet50/output_npu_***`下查看迁移后的脚本。

    将原始脚本文件夹重命名，例如将/root/models/official/vision/image_classification/resnet重命名为resnet_org。

    由于导入库文件依赖原始文件夹结构，要将迁移后的/root/models/resnet50/resnet_npu_\*\*\*重命名为resnet，并拷贝回原始目录，命令示例如下：

    ```bash
    cp -r /root/models/resnet50/resnet_npu_\*\*\* /root/models/official/vision/image_classification/resnet
    ```

## 执行单Device训练

1. 由于原始脚本支持分布式训练，迁移后的脚本中使用了HCCL集合通信接口，则需要在单Device上执行训练前准备单Device的资源信息配置文件。否则请跳过此步。

    （本示例以配置文件的方式设置资源信息，您也可以参见[训练执行（通过环境变量配置资源信息）](../model_training/distributed_training.md#训练执行通过环境变量配置资源信息)通过环境变量的方式设置资源信息。）

    单Device的rank table资源信息配置文件中需包含一个Device资源，文件名举例：rank_table_1p.json，配置文件举例：

    > [!NOTE]说明
    > 不同型号的AI处理器，其rank table文件的配置格式可能有所不同，以下内容仅作为示例参考，如需了解详细rank table配置说明，请参考《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“相关参考 \> 集群信息配置”章节。

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

2. 配置训练进程启动依赖的环境变量。

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh
    
    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
    
    # 当前脚本所在路径，例如：
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export JOB_ID=10086        # 训练任务ID，用户自定义，仅支持大小写字母，数字，中划线，下划线。不建议使用以0开始的纯数字
    export ASCEND_DEVICE_ID=0  # 指定AI处理器的逻辑ID，单P训练也可不配置，默认为0，在0卡执行训练
    export RANK_ID=0           # 指定训练进程在集合通信进程组中对应的rank标识序号，单P训练固定配置为0
    export RANK_SIZE=1         # 指定当前训练进程对应的Device在本集群大小，单P训练固定配置为1
    export RANK_TABLE_FILE=/root/rank_table_1p.json # 如果用户原始训练脚本中使用了hvd接口或tf.data.Dataset对象的shard接口，需要配置，否则无需配置，需要注意rank table中的参数“device_id”优先级高于环境变量“ASCEND_DEVICE_ID”。
    ```

3. 执行训练脚本启动训练进程：

    ```bash
    python3 /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
    ```

4. 检查训练是否跑通。

    ![](../figures/auto_migration_result.png)

## 执行分布式训练

下面以两个Device为例，说明如何使用迁移后的脚本在NPU上执行分布式训练。

1. 准备包含两个Device的rank table资源信息配置文件，文件名举例：rank_table_2p.json，配置文件举例：

    > [!NOTE]说明
    > 不同型号的AI处理器，其rank table文件的配置格式可能有所不同，以下内容仅作为示例参考，如需了解详细rank table配置说明，请参考《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“相关参考 \> 集群信息配置”章节。

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
                            }, 
                            { 
                             "device_id":"1", 
                             "device_ip":"192.168.1.9", 
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

2. 在不同的shell窗口依次启动不同的训练进程。

    启动训练进程0：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh
    
    # 当前脚本所在路径，例如：
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export RANK_ID=0 
    export RANK_SIZE=2 
    export RANK_TABLE_FILE=/home/test/rank_table_2p.json 
    python3 /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
    ```

    启动训练进程1：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh
    
    # 当前脚本所在路径，例如：
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export ASCEND_DEVICE_ID=1 
    export RANK_ID=1 
    export RANK_SIZE=2 
    export RANK_TABLE_FILE=/home/test/rank_table_2p.json 
    python3 /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
    ```

    > [!NOTE]说明
    > 除了以上方式，您还可以通过自定义启动脚本通过循环方式依次启动多个训练进程，请参考[样例链接](https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow/built-in/nlp/BertBase_Google_ID0631_for_TensorFlow/test/train_performance_8p.sh)。
