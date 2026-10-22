# Training with a Single Device

## Overview

This section details how to run a ported TensorFlow training script on a single device. If model porting is not performed, you can obtain a training script that has been ported and adapted from  [https://gitee.com/ascend/modelzoo](https://gitee.com/ascend/modelzoo)  to experience the training process.

> [!CAUTION]NOTICE
> Each device corresponds to a training process. It is not supported to run multiple training processes on a single device.

## Prerequisites

- You have prepared a TensorFlow training script and a matched dataset.
- **If HCCL collective communication interfaces are used in the training script**, you need to configure the device resource information before training by using the configuration file \(ranktable file\) or environment variables.  **You only need to configure the current device resource for this single-device training**  and start the training process. This section does not describe the procedure. For details, see  [Distributed Training with Multiple Devices](distributed_training.md).

    If you use the ranktable file, set the distributed environment variables  **RANK_ID**  to  **0**  and  **RANK_SIZE**  to  **1**.

## Procedure

1. Configure the environment variables required for starting the training process.

     In addition, the following configurations are required:

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export JOB_ID=10087        # User-defined training job ID. Only letters, digits, hyphens (-), and underscores (_) are supported. You are advised not to use a numeric ID starting with 0.
    export ASCEND_DEVICE_ID=0  # Logical ID of the Ascend AI Processor; optional in single-device training, defaults to 0.
    ```

2. \(Optional\) Configure environment variables for auxiliary functions.

   - Enable computational graph dump by setting the corresponding environment variable before starting the training script to facilitate fault locating.

     ```bash
     export DUMP_GE_GRAPH=2                  # 1: dumps all; 2: dumps without data such as weights; 3: dumps only the network structure.
     export DUMP_GRAPH_PATH=/home/dumpgraph  # Dump path. By default, graphs are dumped to the script execution directory.
     ```

     After the training job is started, several dump graph files are generated in the path `${DUMP_GRAPH_PATH}/pid_${pid}_deviceid_${deviceid}`, including the `.pbtxt` and `.txt` files. Given the large number and sizes of dump files, dump can be skipped if there is no fault locating need.

   - If you want the files generated during program compilation and execution to be flushed to a unified storage directory, you can use the environment variables ASCEND_CACHE_PATH and ASCEND_WORK_PATH to set the paths for storing shared files and process-exclusive files, respectively.

     ```bash
     export ASCEND_CACHE_PATH=/repo/task001/cache
     export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
     ```

    > [!NOTE]NOTE
    > Before setting the environment variables, run the env command to check whether ASCEND_CACHE_PATH and ASCEND_WORK_PATH exist. It is recommended that all functions use the same planned path.

3. Run your training script to start the training process.

    **python3 /home/xxx.py**

## Training Result Check

1. Check that the training process is normal and the loss is converged.

    ![](../figures/single_device_loss_convergence.png)

2. After training, find the following directories and files:
    - **model**  directory: stores checkpoint files and model files. Whether to generate this directory depends on the script implementation. If  **saver = tf.train.Saver\(\)**  and  **saver.save\(\)**  are used in the training script to save the model, files similar to the following are generated:

        ![](../figures/train_result.png)

    - **kernel_meta**: stores the .o and .json files of operators on the condition that  **op_debug_level**  is set to  **3**  in the training script. The files can be used for subsequent fault locating.

## Troubleshooting

If the script execution fails, analyze and locate the fault based on the following logs:

Host log:  **$HOME/ascend/log/run/plog/plog-pid_\*.log**, where  **$HOME**  indicates the root directory of the user on the host.

Device log:  **$HOME/ascend/log/run/device-id/device-pid_\*.log**

You can identify the error module and determine the cause by using ERROR-level logs.

![](../figures/error_log.png)

**Table  1**  Fault location techniques

| Module Name | Error | Solution |
| --- | --- | --- |
| System error | Environment and version mismatch | Check the version mapping and system installation. |
| GE | GE graph build or verification error | Specific error causes are provided for verification errors. You only need to modify the network script as prompted. |
| Runtime | Initialization or graph execution failure due to an environment exception | If initialization fails, check the environment configuration and whether the environment is preempted. |
