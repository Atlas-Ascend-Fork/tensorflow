# Training with a Single Device

This section details how to run a ported TensorFlow training script on a single device.

> [!CAUTION]NOTICE
> Each device corresponds to a training process. It is not supported to run multiple training processes on a single device.

## Prerequisites

- You have prepared a TensorFlow training script and a matched dataset.
- **If HCCL APIs are used in the training script**, you need to configure the device resource before training by using the configuration file \(rank table file\) or environment variables. You only need to configure the current device resource for this single-device training and start the training process. This section does not describe the procedure. For details, see  [Distributed Training with Multiple Devices](distributed_training.md).

    If you use the rank table file, set the distributed environment variables  **RANK_ID**  to  **0**  and  **RANK_SIZE**  to  **1**.

## Procedure

1. Configure the environment variables required for starting the training process.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export JOB_ID=10087        # User-defined training job ID. Only letters, digits, hyphens (-), and underscores (_) are supported. You are advised not to use a number starting with 0.
    export ASCEND_DEVICE_ID=0  # Logical ID of the AI processor, optional in single-device training and defaulted to 0, indicating that training is performed on device 0.
    ```

2. (Optional) Configure environment variables for auxiliary functions.
    - Enable computational graph dump by setting the corresponding environment variable before starting the training script to facilitate fault locating.

        ```bash
        export DUMP_GE_GRAPH=2                  # 1: dumps all; 2: dumps without data such as weights; 3: dumps only the network structure.
        export DUMP_GRAPH_PATH=/home/dumpgraph  # Specify the path for storing dump graph files by using this environment variable.
        ```

        After the training job is started, several dump graph files are generated in the path  **\$\{DUMP_GRAPH_PATH\}/pid_$\{pid\}_deviceid_$\{deviceid\}**, including the .pbtxt and .txt files. Given the large number and sizes of dump files, dump can be skipped if there is no fault locating need.

    - If you want the files generated during program compilation and execution to be flushed to a unified storage directory, you can use the environment variables  **ASCEND_CACHE_PATH**  and  **ASCEND_WORK_PATH**  to set the paths for storing shared files and process-exclusive files, respectively.

        ```bash
        export ASCEND_CACHE_PATH=/repo/task001/cache
        export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
        ```

        For details about the restrictions on the usage of the environment variables  **ASCEND_CACHE_PATH**  and  **ASCEND_WORK_PATH**  and the description of the flushed files, see  Installation and Configuration  in  [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

        > [!NOTE]NOTE
        > Before setting the environment variables, run the  **env**  command to check whether  **ASCEND_CACHE_PATH**  and  **ASCEND_WORK_PATH**  exist. It is recommended that all functions use the same planned path.

3. Run your training script to start the training process.

    ```bash
    python3 /home/xxx.py
    ```

## Training Result Check

1. Check that the training process is normal and the loss is converged.

   ![](../figures/single_device_1.png)

2. After training, find the following directories and files:
    - **model**  directory: stores checkpoint files and model files. Whether to generate this directory depends on the script implementation. If  **saver = tf.train.Saver\(\)**  and  **saver.save\(\)**  are used in the training script to save the model, files similar to the following are generated:

      ![](../figures/single_device_2.png)

    - **kernel_meta**  directory: stores the operator .o and .json files, which can be used for subsequent fault locating. These files do not exist in the default directory. You can modify the training script and pass the value  **3**  to the running parameter  **op_debug_level**  to retain the .o and .json files.

## Troubleshooting

If the script execution fails, analyze and locate the fault based on the following logs:

Path of run logs generated when the app is running on the host:  **\$HOME/ascend/log/run/plog/plog-pid_\*.log**.

Path of the run logs generated when the app is running on the device:  **\$HOME/ascend/log/run/device-id/device-pid_\*.log**.

**$HOME**  indicates the root directory of the user on the host.

You can identify the error module and determine the cause by using ERROR-level logs.

![](../figures/error_log.png)

The following table describes the fault locating process.

| Module Name | Error | Solution |
| --- | --- | --- |
| System error | Environment and version mismatch | Check the version mapping and system installation. |
| GE | GE graph build or verification error | Specific error causes are provided for verification errors. You only need to modify the network script as prompted. |
| Runtime | Initialization or graph execution failure due to an environment exception | If initialization fails, check the environment configuration and whether the environment is occupied by other processes. |
