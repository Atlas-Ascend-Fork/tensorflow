# Distributed Training with Multiple Devices

## Distributed Training Overview

Before starting distributed training across multiple processes, you need to configure the resource information of  AI processors that participate in the distributed training.

Use either of the following methods:

- Setting the ranktable configuration file together with environment variables such as  **RANK_TABLE_FILE**  and  **RANK_ID**

    For details about how to configure resource information and start the training process, see  [Training Execution \(Setting the Configuration File\)](#training-execution-configuring-resources-via-the-rank-table).

- Setting environment variables

    For details about how to configure resource information and start the training process, see  [Training Execution \(Setting Environment Variables\)](#training-execution-configuring-resources-via-environment-variables).

For details about the restrictions on the usage of the environment variables  **ASCEND_CACHE_PATH**  and  **ASCEND_WORK_PATH**  and the description of the flushed files, see  "Installation"  in  _[Environment Variables](https://hiascend.com/document/redirect/CannCommunityEnvRef)_.

> [!NOTE]NOTE
>
> Before setting the environment variables, run the **env** command to check whether **ASCEND_CACHE_PATH** and **ASCEND_WORK_PATH** exist. It is recommended that all functions use the same planned path.

**Before performing distributed training, pay attention to the following points:**

1. Atlas training products: In single-server scenarios, the number of  Atlas training products that participate in collective communication can be 1, 2, 4, or 8. In addition, devices 0 to 3 and devices 4 to 7 form separate networks. When two or four devices are used for training, cross-network clusters cannot be created. In server cluster scenarios, the number of  Ascend AI Processors that participate in collective communication can only be 1 ×  _n_, 2 ×  _n_, 4 ×  _n_, or 8 ×  _n_  \(_n_  is the number of servers participating in training\). If  _n_  is an exponential multiple of 2, the cluster performance is the best. Therefore, this mode is recommended for cluster networking.
2. Atlas A2 products: In single-server scenarios, the number of  Ascend AI Processors that participate in collective communication is not limited. In server cluster scenarios, the number of  Ascend AI Processors that participate in collective communication must be \(1 to 8\) ×  _n_  \(_n_  is the number of servers participating in training\). It is recommended that each server should have the same number of  Ascend AI Processors that participate in collective communication. Otherwise, the performance deteriorates.
3. Atlas A3 products: It is recommended that each supernode should have the same number of servers and each server should have the same number of  Ascend AI Processors. Otherwise, the performance deteriorates.
4. One device corresponds to one training process. It is not supported to run multiple training processes on a single device.
5. The successfully ported TensorFlow training script and the corresponding dataset are ready.
6. When performing training on multiple devices, ensure that the models executed on different devices are the same. Otherwise, the service fails to be executed. For details, see [How Do I Fix Application Errors Caused by Model Execution on Multiple Devices?](../faq/multi-device_model_error.md).

## Training Execution \(Configuring Resources via the Rank Table\)

You can configure the NPU resources for collective communication in the rank table file, and specify the NPU resources to use when starting the training process.

The rank table is in JSON format and records the information of all NPUs involved in collective communication. You can prepare the rank table resource configuration file as described in  "Reference" \> "Cluster Information Configuration"  of  _[HCCL User Guide](https://hiascend.com/en/document/redirect/CannCommunityHcclUg)_.

### Single-Server Multi-Device Scenario

When performing training on multiple devices, ensure the training process is initiated on each participating device.

Assume that there is only one AI Server node and eight devices on the node. You can perform the following steps to construct a startup script to cyclically start the training process on each device.

1. Construct a startup script named  **tf_start_8p.sh**  as follows.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export RANK_SIZE=8
    export RANK_TABLE_FILE=/home/test/rank_table_8p.json    # Path of the rank table resource configuration file. Replace it with the actual path.
    export JOB_ID=10087      # User-defined task ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

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

Assume that there are two AI Server nodes involved in distributed training and each AI Server node has eight devices. You can perform the following steps to construct a startup script to cyclically start the training process on each device.

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
    export JOB_ID=10087  # User-defined task ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

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

        After the training job is started, several dump graph files are generated in the path  **$\{DUMP_GRAPH_PATH\}/pid_$\{pid\}_deviceid_$\{deviceid\}**, including the .pbtxt and .txt files. Given the large number and sizes of dump files, dump can be skipped if there is no fault locating need.

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

## Training Execution \(Configuring Resources Via Environment Variables\)

Before training, you need to configure the resource information of  Ascend AI Processors that participate in the cluster training. You can set environment variables described in this section to initialize the collective communication component.

The following products support resource information configuration using environment variables:

Atlas A2 products

Atlas training products

### Configuration Description

Set the following environment variables on every AI server node where training is performed to configure resource information. The following is an example:

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
- **HCCL_SOCKET_FAMILY**: \(Optional\) IP version used by the communication NIC on the device.  **AF_INET**  indicates that IPv4 is used, and  **AF_INET6**  indicates that IPv6 is used. By default, IPv4 is used preferentially.

**NOTE**:

- If the IP specified by the environment variable  **HCCL_SOCKET_FAMILY**  does not match the obtained NIC information, use the actual NIC information.
   For example, if  **HCCL_SOCKET_FAMILY**  is set to  **AF_INET6**  but only IPv4 NICs are available on the device, IPv4 will be used instead.
- When the preceding environment variables are used to configure cluster information,  **RANK_TABLE_FILE**,  **RANK_ID**, and  **RANK_SIZE**  cannot exist.
- For  Atlas A2 products, if the service is deployed in a single-device multi-process scenario, configure the communication ports used by HCCL on the NPU through the environment variable  **HCCL_NPU_SOCKET_PORT_RANGE**. Otherwise, port conflicts may occur. Note that running multiple processes may increase resource overhead and affect communication performance. Configuration example:

    ```bash
    export HCCL_NPU_SOCKET_PORT_RANGE="auto"
    ```

### Example

Assume a distributed training scenario with 2 AI Server nodes and 16 devices in total, where each AI Server node contains 8 devices. Before starting training processes on each device, configure the following environment variables in the corresponding shells to configure resource information.

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

#### Single-Server Multi-Device Scenario

When performing training on multiple devices, ensure the training process is initiated on each participating device.

Assume that there is only one AI Server node and eight devices on the node. You can construct a startup script to cyclically start the training process on each device.

1. Construct a startup script named  **tf_start_8p.sh**  as follows.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export JOB_ID=10087  # User-defined task ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

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

Assume that there are two AI Server nodes involved in distributed training and each AI Server node has eight devices. You can perform the following steps to construct a startup script to cyclically start the training process on each device.

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
    export JOB_ID=10087  # User-defined task ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

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
