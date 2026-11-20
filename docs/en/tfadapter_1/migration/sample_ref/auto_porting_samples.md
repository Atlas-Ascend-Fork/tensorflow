# Automated Porting Samples

## Overview

The following describes how to use the tool to port a ResNet-50 network.

## Model and Dataset Downloads

1. Download a ResNet-50 model from GitHub.

    ```bash
    git clone -b r1.13.0 https://github.com/tensorflow/models.git
    ```

    Assume that the original model is downloaded to the  **/root/models**  directory. You can view the downloaded ResNet-50 script in the  **/root/models/official/resnet/**  directory.

    ![](../figures/resnet50.png)

2. Download a dataset.

    Run the following commands to get a dataset. You can find more details at:  [https://github.com/tensorflow/models/blob/r1.13.0/official/resnet/README.md](https://github.com/tensorflow/models/blob/r1.13.0/official/resnet/README.md)

    ```bash
    cd /root/models/official/resnet/
    python cifar10_download_and_extract.py
    export PYTHONPATH="$PYTHONPATH:/root/models"
    ```

    By default, the dataset is downloaded to the  **/tmp/cifar10_data**  directory.

## Automated Model Porting

1. Install the tool dependencies in the operating environment.

    ```bash
    pip3 install pandas
    pip3 install xlrd==1.2.0
    pip3 install openpyxl
    pip3 install tkintertable
    pip3 install google_pasta
    ```

2. Perform automated porting using the porting tool.
    1. Go to the directory where the porting tool is located.

        ```bash
        cd  ${TFPLUGIN_INSTALL_PATH}/npu_bridge/convert_tf2npu/
        ```

        $\{TFPLUGIN_INSTALL_PATH\}  is the installation path of the TF Adapter package.

    2. Execute script porting.
        - To perform training with a single device, run the following command:

            ```bash
            python3 main.py -i /root/models/official/resnet/ -o /root/models/official/ -r /root/models/official/
            ```

            - **-i**: specifies the path for storing the original script to be ported.
            - **-o**: specifies the path for storing the ported script. The path cannot be a subdirectory of the original script path.
            - **-r**: specifies the path for storing the generated porting report.

        - To perform distributed training, run the following command:

            ```bash
            python3 main.py -i /root/models/official/resnet/ -o /root/models/official/ -r /root/models/official/ -d tf_strategy
            ```

            In the preceding command,  **-d**  specifies the distribution policy used by the original script, and  **tf_strategy**  indicates the  **tf.distribute.Strategy**  policy.

3. View the porting report in the  **/root/models/official/report_npu_\*\*\***  directory.

4. Replace the original script in the  **/root/models/official/resnet**  directory with the ported script.

    The ported script is stored in the  **/root/models/official/resnet_npu_\*\*\***  directory.

    Before training, copy the scripts in the  **resnet_npu_xxx**  directory to the  **resnet**  directory to ensure proper script execution. The following provides an example:

    1. Back up the original script in the  **resnet**  directory.
    2. Replace the original script with the ported script, for example:

        cp /root/models/official/resnet_npu_\*\*\*/\*.py /root/models/official/resnet

## Training with a Single Device

1. The original script supports distributed training, and the ported script contains the HCCL APIs. Therefore, you need to configure the resource information of a device before training.

    In this sample, the rank table file is used to configure resource information. The rank table file must contain one device resource. Assume that the file is named  **rank_table_1p.json**. The following provides a file template.

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

    For details about the rank table configuration file, see  Reference \> Cluster Information Configuration  in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclug_000001.html).

2. Configure the environment variables required for starting the training process.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # Script directory, for example:
    export PYTHONPATH="$PYTHONPATH:/root/models"
    export JOB_ID=10087        # User-defined training job ID. Only letters, digits, hyphens (-), and underscores (_) are supported. You are advised not to use a number starting with 0.
    export ASCEND_DEVICE_ID=0  # Logical ID of the AI processor, optional in single-device training and defaulted to 0, indicating that training is performed on device 0.
    export RANK_ID=0           # Rank ID of a training process in the collective communication process group. Fixed at 0 in single-device training.
    export RANK_SIZE=1         # Rank size of a device corresponding to the current training process in the cluster. Fixed at 1 in single-device training.
    export RANK_TABLE_FILE=/root/rank_table_1p.json # This parameter needs to be configured only when the hvd API or the shard API of the tf.data.Dataset object is used in the original training script. Note that the device_id parameter in rank_table takes precedence over the environment variable ASCEND_DEVICE_ID.
    ```

3. Run your training script to start the training process.

    **python3 /root/models/official/resnet**/**cifar10_main.py**

4. Check that the training process is normal and the loss is converged.

    ![](../figures/sample_result1.png)

5. After training, find the checkpoint files in  **tmp/cifar10_model**.

    ![](../figures/sample_result2.png)

## Distributed Training with Two Devices

1. Prepare a 2-device rank table resource configuration file. Assume that the file is named  **rank_table_2p.json**. The following provides a file template.

    ```json
    {
    "server_count":"1",
    "server_list":
    [
       {
            "device":[
                           {
                            "device_id":"0",     // This configuration takes precedence over the environment variable ASCEND_DEVICE_ID.
                            "device_ip":"192.168.1.8",
                            "rank_id":"0"
                            },
                            {
                             "device_id":"1",
                             "device_ip":"192.168.1.9",   // The two devices must be in the same network segment. Here, devices 0 and 1 are in the same network segment.
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

    For details about the rank table configuration file, see  Reference \> Cluster Information Configuration  in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclug_000001.html).

2. Start the training processes in different shells.

    Start training process 0.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # Script directory, for example:
    export PYTHONPATH=/root/models:$PYTHONPATH
    export JOB_ID=10087

    export RANK_ID=0
    export RANK_SIZE=2
    export RANK_TABLE_FILE=/root/rank_table_2p.json
    python3 /root/models/official/resnet/cifar10_main.py
    ```

    Start training process 1.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # Script directory, for example:
    export PYTHONPATH=/root/models:$PYTHONPATH
    export JOB_ID=10087

    export RANK_ID=1
    export RANK_SIZE=2
    export RANK_TABLE_FILE=/root/rank_table_2p.json
    python3 /root/models/official/resnet/cifar10_main.py
    ```

3. After the training is complete, the log information of each device is as follows, indicating that the training is complete. You can check whether the loss is converged.

    ![](../figures/sample_result3.png)

    The checkpoint files are generated in  **tmp/cifar10_model**.
