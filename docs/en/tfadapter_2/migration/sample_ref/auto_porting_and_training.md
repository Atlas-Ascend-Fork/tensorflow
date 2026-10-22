# Automated Porting and Training

## Overview

The following describes how to use the tool to port a ResNet-50 network.

## Model and Dataset Downloads

1. Download a ResNet-50 model from GitHub.

    **git clone -b r2.6.0 <https://github.com/tensorflow/models.git>**

    If the source code is downloaded to the  **/root/models**  directory, you can find the downloaded script in the  **/root/models/official/vision/image_classification/resnet/**  directory.

    ![](../figures/resnet50_download.png)

2. Download a dataset.

    Download the ImageNet2012 dataset and transform data into TFRecord by using the  **imagenet_to_gcs.py**  script. For details, click  [here](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy).

    Save the processed dataset to the  **/root/models/data/imagenet_TF/**  directory.

## Automated Model Porting

1. Before porting, manually add dataset sharding logic as required in  [Constraints](../script_migration/automated_porting.md#tool-overview).

    ```python
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    import npu_device as npu
    # Shard logic added by the NPU. The dataset and global batch will be sharded based on the number of clusters.
    dataset, batch_size = npu.distribute.shard_and_rebatch_dataset(dataset, batch_size)
    #if input_context:
    #  logging.info(
    #      'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
    #      input_context.input_pipeline_id, input_context.num_input_pipelines)
    #  dataset = dataset.shard(input_context.num_input_pipelines,
    #                          input_context.input_pipeline_id)
    ```

2. Install the tool dependencies in the operating environment.

    ```bash
    pip3 install pandas
    pip3 install openpyxl
    pip3 install google_pasta
    ```

3. Perform automated porting using the porting tool.
    1. Run the following command to navigate to the tool directory:

       ```bash
       cd ${TFPLUGIN_INSTALL_PATH}/npu_device/convert_tf2npu/
       ```

       $\{TFPLUGIN_INSTALL_PATH\}  indicates the installation path of the TF Adapter package.

    2. Execute script porting.
        - To perform training with a single device, run the following command:

            ```bash
            python3 main.py -i /root/models/official/vision/image_classification/resnet/ -o /root/models/resnet50/ -r /root/models/resnet50/ -m /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
            ```

        - To perform distributed training, run the following command:

            ```bash
            python3 main.py -i /root/models/official/vision/image_classification/resnet/ -o /root/models/resnet50/ -r /root/models/resnet50/ -m /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py -d tf_strategy
            ```

            In the preceding command,  **-d**  specifies the distributed policy used by the original script, and  **tf_strategy**  indicates the  **tf.distribute.Strategy**  policy.

4. Check the porting report in the  **/root/models/resnet50/report_npu_\*\*\***  directory.
5. Check the resultant script in the  **/root/models/resnet50/output_npu_\*\*\***  directory.

    Rename the original script folder. For example, change  **/root/models/official/vision/image_classification/resnet**  to  **resnet_org**.

    The library import depends on the original directory structure. As such, you need to rename the  **/root/models/resnet50/resnet_npu_\*\*\***  file  **resnet**  and copy it back to the original directory.

    ```bash
    cp -r /root/models/resnet50/resnet_npu_\*\*\* /root/models/official/vision/image_classification/resnet
    ```

## Training with a Single Device

1. The original script supports distributed training, and the ported script uses the HCCL APIs. Therefore, you need to prepare the resource information configuration file of a single device before training with a single device. Skip this step if not that case.

    \(In this example, the resource information is set using the configuration file. You can also use the method described in  [Training Execution \(Setting Environment Variables\)](../model_training/distributed_training.md#training-execution-configuring-resources-via-environment-variables).\)

    Ensure that the single-device rank table resource configuration file contains only one device resource. Assume that the file is named  **rank_table_1p.json**. The following provides a file template.

    > [!NOTE]NOTE
    > The rank table file format may vary with the  AI processor  model. The following information is for reference only. For details about the rank table configuration, see  Reference \> Cluster Information Configuration  in  [Huawei Collective Communication Library \(HCCL\)](https://hiascend.com/en/document/redirect/CannCommunityHcclUg).

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

2. Configure the environment variables required for starting the training process.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh
    
    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
    
    # Script directory, for example:
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export JOB_ID=10086        # User-defined training job ID. Only letters, digits, hyphens (-), and underscores (_) are supported. You are advised not to use a number starting with 0.
    export ASCEND_DEVICE_ID=0  # Logical ID of the AI processor, optional in single-device training and defaulted to 0, indicating that training is performed on device 0.
    export RANK_ID=0           # Rank ID of a training process in the collective communication process group. Fixed at 0 in single-device training.
    export RANK_SIZE=1         # Rank size of a device corresponding to the current training process in the cluster. Fixed at 1 in single-device training.
    export RANK_TABLE_FILE=/root/rank_table_1p.json # This parameter needs to be configured only when the shard API of the tf.data.Dataset object or the hvd API is used in the original training script. Note that the device_id parameter in the rank table takes precedence over the environment variable ASCEND_DEVICE_ID.
    ```

3. Run your training script to start the training process.

    ```bash
    python3 /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
    ```

4. Check whether the training is successful.

    ![](../figures/auto_migration_result.png)

## Distributed Training with Multiple Devices

The following uses 2-device training as an example to describe how to use the ported script to perform distributed training on the NPU.

1. Prepare a 2-device rank table resource configuration file. Assume that the file is named  **rank_table_2p.json**. The following provides a file template.

    > [!NOTE]NOTE
    > The rank table file format may vary with the  AI processor  model. The following information is for reference only. For details about the rank table configuration, see  Reference \> Cluster Information Configuration  in  [Huawei Collective Communication Library \(HCCL\)](https://hiascend.com/en/document/redirect/CannCommunityHcclUg).

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

2. Start the training processes in different shells.

    Start training process 0.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh
    
    # Script directory, for example:
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export RANK_ID=0 
    export RANK_SIZE=2 
    export RANK_TABLE_FILE=/home/test/rank_table_2p.json 
    python3 /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
    ```

    Start training process 1.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh
    
    # Script directory, for example:
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export ASCEND_DEVICE_ID=1 
    export RANK_ID=1 
    export RANK_SIZE=2 
    export RANK_TABLE_FILE=/home/test/rank_table_2p.json 
    python3 /root/models/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py
    ```

    > [!NOTE]NOTE
    > Alternatively, you can also customize a startup script to start multiple training processes through loops. For details, click  [here](https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow/built-in/nlp/BertBase_Google_ID0631_for_TensorFlow/test/train_performance_8p.sh).
