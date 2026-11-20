# Distributed Training with Multiple Devices

## Precautions

Before performing distributed training, refer to this section for some precautions.

Before starting distributed training across multiple processes, you need to configure the resource information of  AI processors that participate in the distributed training.

Currently, resource information can be configured using configuration files or environment variables. **You can choose either of them, but they cannot be used together**.

- Using configuration files. The resource configuration file, known as a rank table file, is used alongside environment variables such as **RANK_TABLE_FILE** and **RANK_ID**.
- Setting environment variables

  This method is supported only by the following products:
  - Atlas A2 products
  - Atlas training products

For details about the restrictions on the usage of the environment variables **ASCEND_CACHE_PATH** and **ASCEND_WORK_PATH** mentioned below and the description of the flushed files, see  Installation and Configuration  in  [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

> [!NOTE]NOTE
> Before setting the environment variables, run the **env** command to check whether **ASCEND_CACHE_PATH** and **ASCEND_WORK_PATH** exist. It is recommended that all functions use the same planned path.

**Before performing distributed training, pay attention to the following points:**

1. You have prepared a TensorFlow training script and a matched dataset.
2. When performing training on multiple devices, ensure that the models executed on different devices are the same. Otherwise, the service fails to be executed. For details, see  [How Do I Fix Application Errors Caused by Model Execution on Multiple Devices?](../faq/multi-device_model_error.md).
3. Atlas training products: In single-server scenarios, the number of  AI processors that participate in collective communication can be 1, 2, 4, or 8. In addition, devices 0 to 3 and devices 4 to 7 form separate networks. When two or four devices are used for training, cross-network clusters cannot be created. In server cluster scenarios, the number of  AI processors that participate in collective communication can only be 1 x  _n_, 2 x  _n_, 4 x  _n_, or 8 x  _n_  \(_n_  is the number of servers participating in training\). If  _n_  is an exponential multiple of 2, the cluster performance is the best. Therefore, this mode is recommended for cluster networking.
4. Atlas A2 products: In single-server scenarios, the number of  AI processors that participate in collective communication is not limited. In server cluster scenarios, the number of  AI processors that participate in collective communication must be \(1 to 8\) x  _n_  \(_n_  is the number of servers participating in training\). It is recommended that each server should have the same number of  AI processors that participate in collective communication. Otherwise, the performance deteriorates.
5. Atlas A3 products: It is recommended that each supernode should have the same number of servers and each server should have the same number of  AI processors. Otherwise, the performance deteriorates.
6. Each device corresponds to a training process. It is not supported to run multiple training processes on a single device.

## Training Execution (Configuring Resources via the Rank Table)

You can configure the NPU resources for collective communication in the rank table file, and specify the NPU resources to use when starting the training process.

### Preparing the Rank Table Resource Configuration File

The rank table is in JSON format and records the information of all NPUs involved in collective communication. You can prepare the rank table resource configuration file as described in  Reference \> Cluster Information Configuration  of  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclug_000001.html).

### Single-Server Multi-Device Scenario

When performing training on multiple devices, ensure the training process is initiated on each participating device.

Assume that there is only one AI Server node and eight devices on the node. You can perform the following steps to construct a startup script to cyclically start the training process on each device.

1. Construct a startup script named **tf_start_8p.sh** as follows.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export RANK_SIZE=8
    export RANK_TABLE_FILE=/home/test/rank_table_8p.json    # Path of the rank table resource configuration file. Replace it with the actual path.
    export JOB_ID=10087      # User-defined job ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

    for((RANK_ID=0;RANK_ID<$((RANK_SIZE));RANK_ID++));
    do
        export RANK_ID=$RANK_ID
        export ASCEND_DEVICE_ID=$RANK_ID
        # Execute the training script. Replace the training script path, name, and other input parameters as required.
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    **\(Optional\)**  Before starting the training process, configure environment variables of the following auxiliary functions.

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

2. Run your script to start the training process.

    ```bash
    bash tf_start_8p.sh
    ```

### Multi-Server Multi-Device Scenario

When performing training on multiple devices, ensure the training process is initiated on each participating device.

Assume that there are two AI server nodes involved in distributed training and each AI Server node has eight devices. You can perform the following steps to construct a startup script to cyclically start the training process on each device.

1. Construct a startup script named  **tf_start_16p.sh**  as follows.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # Obtain input parameters.
    for para in $*
    do
        if [[ $para == --server_index* ]];then
            server_index=`echo ${para#*=}`
        elif [[ $para == --devices_num* ]];then
         devices_num=`echo ${para#*=}`
        elif [[ $para == --servers_num* ]];then
            servers_num=`echo ${para#*=}`
        fi
    done

    rank_size=${devices_num}
    linux_num=$servers_num
    export RANK_SIZE=`awk 'BEGIN{printf "%.0f\n",'${devices_num}'*'${linux_num}'}'`
    export RANK_TABLE_FILE=/home/test/rank_table.json   # Path of the rank table resource configuration file. Replace it with the actual path.
    export JOB_ID=10087  # User-defined job ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

    for((RANK_ID=$((rank_size*server_index));RANK_ID<$((((server_index+1))*rank_size));RANK_ID++));
    do
        # Set environment variables.
        export RANK_ID=$RANK_ID
        export ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
        # Execute the training script. Replace the training script path, name, and other input parameters as required.
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    **\(Optional\)**  Before starting the training process, configure environment variables of the following auxiliary functions.

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

2. Run your script to start the training process.

    ```bash
    # Start the training process on node 0.
    bash tf_start_16p.sh --server_index=0 --devices_num=8 --servers_num=2
    # Start the training process on node 1.
    bash tf_start_16p.sh --server_index=1 --devices_num=8 --servers_num=2
    ```

## Training Execution (Configuring Resources Via Environment Variables)

### Configuring Resource Information

The following  Products support resource information configuration using environment variables:

- Atlas A2 products
- Atlas training products

To configure resource information, set the following environment variables on every AI server node where training is needed. The following is an example:

```bash
export CM_CHIEF_IP=192.168.1.1
export CM_CHIEF_PORT=6000
export CM_CHIEF_DEVICE=0
export CM_WORKER_SIZE=8
export CM_WORKER_IP=192.168.0.1
export HCCL_SOCKET_FAMILY=AF_INET
```

- **CM_CHIEF_IP**: host listening IP address of the master node, that is, the IP address used to communicate with other nodes. The value must be in the IPv4 or IPv6 format.
- **CM_CHIEF_PORT**: listening port of the master node. The value must be an integer ranging from 0 to 65520. Ensure that the port is not occupied by other processes.
- **CM_CHIEF_DEVICE**: logical ID of the device that collects server cluster information on the master node.

    The value of this environment variable must be an integer within the range of \[0, Maximum number of devices in the server – 1\].

- **CM_WORKER_SIZE**: total number of devices involved in cluster training on the network. The value must be an integer ranging from 0 to 32768.
- **CM_WORKER_IP**: IP address of the NIC used by the current node to communicate with the master node. The value must be in the IPv4 or IPv6 format.
- **HCCL_SOCKET_FAMILY**: \(Optional\) IP version used by the communication NIC on the device.  **AF_INET**  indicates that IPv4 is used, and  **AF_INET6**  indicates that IPv6 is used.  **By default, IPv4 is used preferentially.**

NOTE:

- If the IP specified by the environment variable  **HCCL_SOCKET_FAMILY**  does not match the obtained NIC information, the actual NIC information will be used.

  For example, if  **HCCL_SOCKET_FAMILY**  is set to  **AF_INET6**  but only IPv4 NICs are available on the device, IPv4 will be used instead.
- When the preceding environment variables are used to configure cluster information, RANK_TABLE_FILE, RANK_ID, and RANK_SIZE cannot exist.
- For  Atlas A2 products, if the service is deployed in a single-device multi-process scenario, configure the communication ports used by HCCL on the NPU through the environment variable  **HCCL_NPU_SOCKET_PORT_RANGE**. Otherwise, port conflicts may occur. Running multiple processes may increase resource overhead and affect communication performance. Configuration example:

    ```bash
    export HCCL_NPU_SOCKET_PORT_RANGE="auto"
    ```

Assume that there are two AI server nodes and 16 devices \(that is, eight on each AI server node\) for distributed training. Before starting training processes on each device, configure the following environment variables in the corresponding shells to configure resource information.

- Node 0 is used as the master node, responsible for managing cluster information, resource allocation, and scheduling.

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.1.1
    ```

- Node 1

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.2.1
    ```

### Performing Training

After configuring resources for distributed training using environment variables, you can start the training process by referring to this section.

#### Single-Server Multi-Device Scenario

When performing training on multiple devices, ensure the training process is initiated on each participating device.

Assume that there is only one AI server node and eight devices on the node. You can construct a startup script to cyclically start the training process on each device.

1. Construct a startup script named  **tf_start_8p.sh**  as follows.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export JOB_ID=10087  # User-defined job ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

    for((CURRENT_DEVICE=0;CURRENT_DEVICE<8;CURRENT_DEVICE++));
    do
        export ASCEND_DEVICE_ID=${CURRENT_DEVICE}
        # Execute the training script. Replace the training script path, name, and other input parameters as required.
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    **\(Optional\)**  Before starting the training process, configure environment variables of the following auxiliary functions.

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

2. Run your script to start the training process.

    ```bash
    bash tf_start_8p.sh
    ```

#### Multi-Server Multi-Device Scenario

When performing training on multiple devices, ensure the training process is initiated on each participating device.

Assume that there are two AI server nodes involved in distributed training and each AI Server node has eight devices. You can perform the following steps to construct a startup script to cyclically start the training process on each device.

1. Construct a startup script named  **tf_start_16p.sh**  as follows.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # Obtain input parameters.
    for para in $*
    do
        if [[ $para == --server_index* ]];then
            server_index=`echo ${para#*=}`
        elif [[ $para == --devices_num* ]];then
     devices_num=`echo ${para#*=}`
        fi
    done

    rank_size=${devices_num}
    linux_num=$servers_num
    export JOB_ID=10087  # User-defined job ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

    for((CURRENT_DEVICE=$((rank_size*server_index));CURRENT_DEVICE<$((((server_index+1))*rank_size));CURRENT_DEVICE++));
    do
        export ASCEND_DEVICE_ID=`expr ${CURRENT_DEVICE} - $((rank_size*server_index))`
        # Execute the training script. Replace the training script path, name, and other input parameters as required.
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    **\(Optional\)**  Before starting the training process, configure environment variables of the following auxiliary functions.

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

2. Run your script to start the training process.

    ```bash
    # Start the training process on node 0.
    bash tf_start_16p.sh --server_index=0 --devices_num=8
    # Start the training process on node 1.
    bash tf_start_16p.sh --server_index=1 --devices_num=8
    ```

### Result Description

After distributed training is complete, you can check the execution result and locate faults by referring to this section.

1. Check your script execution result.

    The print result varies according to training scripts. If information similar to the following is displayed on each device for distributed training, the training is complete.

    ![](../figures/distributed_train_1.png)

    When the environment variable  **DUMP_GE_GRAPH**  is enabled, GE dump graph files are generated.

    ```bash
    export DUMP_GE_GRAPH=2
    ```

    If the HcomBroadcast and HcomAllReduce operators are found in the directory of the dumped graph files, the HCCL operators for inter-NPU communication have been properly inserted.

    Dumped graphs from GE：

    ![](../figures/ge_dump.png "dumped-graphs-from-ge")

2. If your script fails to execute, analyze and locate the fault in the same way you do in single-device training.

    You can spot the fault by checking the host log file  **plog_\*.log**  in  **$HOME/ascend/log/run/plog**  where  **$HOME**  is the root directory of the host user.

    If the execution succeeds on a single device but fails on multiple devices, the issue is typically related to collective communication, as shown in the following figure. For details, see section "FAQs" in  [Huawei Collective Communication Library (HCCL)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclug_000001.html).

    ![](../figures/hccl_faq.png)

If the script execution fails, analyze and locate the fault based on the following logs:

Path of run logs generated when the app is running on the host:  **\$HOME/ascend/log/run/plog/plog-pid_\*.log**.

Path of the run logs generated when the app is running on the device:  **\$HOME/ascend/log/run/device-id/device-pid_\*.log**.

**$HOME**  indicates the root directory of the user on the host.

> [!NOTE]NOTE
> For more information, see  [Log Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/logreference/logreference_0001.html).

You can identify the error module and determine the cause by using ERROR-level logs.

Error log example:

![](../figures/error_log.png "error-log-example")

The following table describes the fault locating process.

| Module Name | Error | Solution |
| --- | --- | --- |
| System error | Environment and version mismatch | Check the version mapping and system installation. |
| GE | GE graph build or verification error | Specific error causes are provided for verification errors. You only need to modify the network script as prompted. |
| Runtime | Initialization or graph execution failure due to an environment exception | If initialization fails, check the environment configuration and whether the environment is occupied by other processes. |
