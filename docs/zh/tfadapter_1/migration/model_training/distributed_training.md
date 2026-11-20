# 执行分布式训练

## 使用前须知

执行分布式训练前，首先应参考本节了解一些注意事项。

开发者跨多个进程执行分布式训练时，首先需要配置参与分布式训练的AI处理器的资源信息，然后再启动训练进程。

当前有配置文件和环境变量两种配置资源信息的方式，**开发者可以选择其中任一方式，但两种方式不能混合使用**。

- 通过配置文件的方式，此资源配置文件称为rank table文件，并配合RANK_TABLE_FILE、RANK_ID等环境变量使用。
- 通过环境变量的方式。

  该方式仅支持如下产品：
  - Atlas A2系列产品
  - Atlas训练系列产品

下文中涉及到的环境变量ASCEND_CACHE_PATH与ASCEND_WORK_PATH的使用约束以及落盘文件说明，可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》中的“安装配置相关”章节。

  > [!NOTE]说明
  > 配置这些环境变量前，请使用**env**命令查询ASCEND_CACHE_PATH与ASCEND_WORK_PATH环境变量是否已存在，建议系统各功能使用统一的规划路径。

**执行分布式训练前，请了解如下注意事项：**

1. 已准备好迁移成功的TensorFlow训练脚本和对应数据集。
2. 多Device上执行训练时，请确保不同Device上执行的模型相同，否则业务会执行失败，详细可参见[多Device上执行模型不同导致应用程序出错](../faq/multi-device_model_error.md)。
3. 针对Atlas训练系列产品，单Server场景下，要求实际参与集合通信的AI处理器数目为1、2、4、8，且0-3卡和4-7卡各为一个组网，使用2张卡或4张卡训练时，不支持跨组网创建设备集群；Server集群场景，要求参与集合通信的AI处理器数目只能为1\*n、2\*n、4\*n、8\*n（n为参与训练的Server个数），且n为2的指数倍情况下，集群性能最好，建议用户优先采用此种方式进行集群组网。
4. 针对Atlas A2系列产品，单Server场景，对参与集合通信的AI处理器数量无限制；Server集群场景要求参与集合通信的AI处理器数量为（1~8）\*n（n为参与训练的Server个数）。建议每个Server中参与集合通信的AI处理器数量保持一致，若不一致，会造成性能劣化。
5. 针对Atlas A3系列产品，建议每个超节点中的Server数量一致，每个Server中的AI处理器数量一致，若不一致，会造成性能劣化。
6. 一个Device对应执行一个训练进程，当前不支持多进程在同一个Device上进行训练。

## 训练执行（通过rank table配置资源信息）

开发者可以通过rank table文件配置参与集合通信的NPU资源信息，并在后续启动训练进程时指定使用的NPU资源。

### 准备rank table资源配置文件

rank table为JSON格式，记录了参与集合通信的所有NPU信息，开发者可以参见《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》中的“相关参考 \> 集群信息配置”章节准备rank table资源配置文件。

### 单机多卡场景

多Device上执行训练时，需要依次在每个参与训练的Device上启动训练进程。

假设只有一个AI Server节点，节点上有8个Device，开发者可以参见如下步骤构造启动脚本，循环启动每个Device上的训练进程。

1. 构造启动脚本，假设命名为tf_start_8p.sh，示例如下：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export RANK_SIZE=8
    export RANK_TABLE_FILE=/home/test/rank_table_8p.json    # rank table资源配置文件路径，请根据实际情况替换
    export JOB_ID=10087      # 用户自定义，指定任务ID，可以包含大小写字母、数字、中划线或下划线

    for((RANK_ID=0;RANK_ID<$((RANK_SIZE));RANK_ID++));
    do
        export RANK_ID=$RANK_ID
        export ASCEND_DEVICE_ID=$RANK_ID
        # 执行训练脚本，训练脚本路径、名称及其他输入参数请根据实际情况替换
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    （可选）训练进程启动前，您还可以配置如下辅助功能的环境变量。

    - 为了后续方便定位问题，启动训练脚本前用户也可以通过环境变量使能dump计算图。

        ```bash
        export DUMP_GE_GRAPH=2   # 1：全量dump；2：不含有权重等数据的基本版dump；3：只显示节点关系的精简版dump
        export DUMP_GRAPH_PATH=/home/dumpgraph  # 通过该环境变量指定dump图文件存储路径
        ```

        训练任务启动后，会在\$\{DUMP_GRAPH_PATH\}/pid_$\{pid\}_deviceid_\$\{deviceid\}路径下生成若干dump图文件，包括后缀为“.pbtxt”和“.txt”的dump文件。由于dump的数据文件较多且文件都较大，若非问题定位需要，可以不生成dump图。

    - 若开发者期望程序编译运行过程中产生的文件落盘到统一存储目录，可通过环境变量ASCEND_CACHE_PATH与ASCEND_WORK_PATH分别设置共享文件的存储路径与进程独享文件的存储路径。

        ```bash
        export ASCEND_CACHE_PATH=/repo/task001/cache
        export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
        ```

2. 执行脚本，启动训练进程。

    ```bash
    bash tf_start_8p.sh
    ```

### 多机多卡场景

多Device上执行训练时，需要依次在每个参与训练的Device上启动训练进程。

假设参与分布式训练的AI Server节点数量为2，每个AI Server节点有8个Device，开发者可以参见如下步骤构造启动脚本，循环启动每个Device上的训练进程。

1. 构造启动脚本，假设命名为tf_start_16p.sh，示例如下：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # 获取输入参数
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
    export RANK_TABLE_FILE=/home/test/rank_table.json   # rank table资源配置文件路径，请根据实际情况替换
    export JOB_ID=10087  # 用户自定义，指定任务ID，可以包含大小写字母、数字、中划线或下划线

    for((RANK_ID=$((rank_size*server_index));RANK_ID<$((((server_index+1))*rank_size));RANK_ID++));
    do
        #设置环境变量
        export RANK_ID=$RANK_ID
        export ASCEND_DEVICE_ID=`expr ${RANK_ID} - $((rank_size*server_index))`
        # 执行训练脚本，训练脚本路径、名称及其他输入参数请根据实际情况替换
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    （可选）训练进程启动前，您还可以配置如下辅助功能的环境变量。

    - 为了后续方便定位问题，启动训练脚本前用户也可以通过环境变量使能dump计算图。

        ```bash
        export DUMP_GE_GRAPH=2                  # 1：全量dump；2：不含有权重等数据的基本版dump；3：只显示节点关系的精简版dump
        export DUMP_GRAPH_PATH=/home/dumpgraph  # 通过该环境变量指定dump图文件存储路径
        ```

        训练任务启动后，会在\$\{DUMP_GRAPH_PATH\}/pid_$\{pid\}_deviceid_\$\{deviceid\}路径下生成若干dump图文件，包括后缀为“.pbtxt”和“.txt”的dump文件。由于dump的数据文件较多且文件都较大，若非问题定位需要，可以不生成dump图。

    - 若开发者期望程序编译运行过程中产生的文件落盘到统一存储目录，可通过环境变量ASCEND_CACHE_PATH与ASCEND_WORK_PATH分别设置共享文件的存储路径与进程独享文件的存储路径。

        ```bash
        export ASCEND_CACHE_PATH=/repo/task001/cache
        export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
        ```

2. 执行脚本，启动训练进程。

    ```bash
    # 启动节点0上的训练进程
    bash tf_start_16p.sh --server_index=0 --devices_num=8 --servers_num=2
    # 启动节点1上的训练进程
    bash tf_start_16p.sh --server_index=1 --devices_num=8 --servers_num=2
    ```

## 训练执行（通过环境变量配置资源信息）

### 通过环境变量配置资源信息

通过环境变量配置资源信息的方式仅支持如下产品：

- Atlas A2系列产品
- Atlas训练系列产品

开发者需要在执行训练的每个AI Server节点上分别配置如下环境变量，进行资源信息的配置，示例如下：

```bash
export CM_CHIEF_IP=192.168.1.1
export CM_CHIEF_PORT=6000
export CM_CHIEF_DEVICE=0
export CM_WORKER_SIZE=8
export CM_WORKER_IP=192.168.0.1
export HCCL_SOCKET_FAMILY=AF_INET
```

- CM_CHIEF_IP：Master节点的Host监听IP，即与其他节点进行通信的IP地址，要求为常规IPv4或IPv6格式。
- CM_CHIEF_PORT：Master节点的监听端口，需要配置为整数，取值范围“0～65520”，请确保端口未被其他进程占用。
- CM_CHIEF_DEVICE：Master节点中统计Server端集群信息的Device逻辑ID。

  该环境变量需要配置为整数，取值范围：\[0，Server内的最大Device数量-1\]。

- CM_WORKER_SIZE：用于配置组网中参与集群训练的Device总数量，需要配置为整数，取值范围“0~32768”。
- CM_WORKER_IP：用于配置当前节点与Master进行通信时所用的网卡IP，要求为常规IPv4或IPv6格式。
- HCCL_SOCKET_FAMILY：**此环境变量可选**，用于控制Device侧通信网卡使用的IP协议版本。AF_INET代表使用IPv4协议，AF_INET6代表使用IPv6协议，**缺省时，优先使用IPv4协议**。

说明：

- 如果环境变量“HCCL_SOCKET_FAMILY”指定的IP协议与实际获取到的网卡信息不匹配，则以实际环境上的网卡信息为准。

   例如，环境变量“HCCL_SOCKET_FAMILY”指定为“AF_INET6”，但Device侧只存在IPv4协议的网卡，则实际会使用IPv4协议的网卡。
- 通过以上环境变量的方式配置集群信息时，环境中不能存在环境变量RANK_TABLE_FILE、RANK_ID、RANK_SIZE。
- 针对Atlas A2系列产品，若业务为单卡多进程场景，建议通过环境变量“HCCL_NPU_SOCKET_PORT_RANGE”配置HCCL在NPU侧使用的通信端口，否则可能会导致端口冲突，但需要注意，多进程会对资源开销、通信性能产生一定的影响，配置示例：

   ```bash
   export HCCL_NPU_SOCKET_PORT_RANGE="auto"
   ```

假设执行分布式训练的AI Server节点数量为2，Device数量为16为例，每个AI Server节点有8个Device。启动每个Device上的训练进程前，在对应的shell窗口中配置如下所示环境变量，进行资源信息的配置。

- 节点0，此节点为Master节点，负责集群信息管理、资源分配与调度。

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.1.1
    ```

- 节点1

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.2.1
    ```

### 执行训练

通过环境变量方式配置好参与分布式训练的资源信息后，可参考以下步骤启动训练进程，执行训练。

#### 单机多卡场景

多Device上执行训练时，需要依次在每个参与训练的Device上启动训练进程。

假设只有一个AI Server节点，节点上有8个Device，开发者可以构造启动脚本循环启动每个Device上的训练进程。

1. 构造启动脚本，假设命名为tf_start_8p.sh，示例如下：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export JOB_ID=10087  # 用户自定义，指定任务ID，可以包含大小写字母、数字、中划线或下划线

    for((CURRENT_DEVICE=0;CURRENT_DEVICE<8;CURRENT_DEVICE++));
    do
        export ASCEND_DEVICE_ID=${CURRENT_DEVICE}
        # 执行训练脚本，训练脚本路径、名称及其他输入参数请根据实际情况替换
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    （可选）训练进程启动前，您还可以配置如下辅助功能的环境变量。

    - 为了后续方便定位问题，启动训练脚本前用户也可以通过环境变量使能dump计算图。

        ```bash
        export DUMP_GE_GRAPH=2                  # 1：全量dump；2：不含有权重等数据的基本版dump；3：只显示节点关系的精简版dump
        export DUMP_GRAPH_PATH=/home/dumpgraph  # 通过该环境变量指定dump图文件存储路径
        ```

        训练任务启动后，会在\$\{DUMP_GRAPH_PATH\}/pid_\$\{pid\}_deviceid_\$\{deviceid\}路径下生成若干dump图文件，包括后缀为“.pbtxt”和“.txt”的dump文件。由于dump的数据文件较多且文件都较大，若非问题定位需要，可以不生成dump图。

    - 若开发者期望程序编译运行过程中产生的文件落盘到统一存储目录，可通过环境变量ASCEND_CACHE_PATH与ASCEND_WORK_PATH分别设置共享文件的存储路径与进程独享文件的存储路径。

        ```bash
        export ASCEND_CACHE_PATH=/repo/task001/cache
        export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
        ```

2. 执行脚本，启动训练进程。

    ```bash
    bash tf_start_8p.sh
    ```

#### 多机多卡场景

多Device上执行训练时，需要依次在每个参与训练的Device上启动训练进程。

假设参与分布式训练的AI Server节点数量为2，每个AI Server节点有8个Device，开发者可以参见如下步骤构造启动脚本，循环启动每个Device上的训练进程。

1. 构造启动脚本，假设命名为tf_start_16p.sh，示例如下：

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # 获取输入参数
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
    export JOB_ID=10087  # 用户自定义，指定任务ID，可以包含大小写字母、数字、中划线或下划线

    for((CURRENT_DEVICE=$((rank_size*server_index));CURRENT_DEVICE<$((((server_index+1))*rank_size));CURRENT_DEVICE++));
    do
        export ASCEND_DEVICE_ID=`expr ${CURRENT_DEVICE} - $((rank_size*server_index))`
        # 执行训练脚本，训练脚本路径、名称及其他输入参数请根据实际情况替换
        nohup python3 /home/test/main.py > /home/test/train_$ASCEND_DEVICE_ID.log 2>&1 &
    done
    ```

    （可选）训练进程启动前，您还可以配置如下辅助功能的环境变量。

    - 为了后续方便定位问题，启动训练脚本前用户也可以通过环境变量使能dump计算图。

        ```bash
        export DUMP_GE_GRAPH=2                  # 1：全量dump；2：不含有权重等数据的基本版dump；3：只显示节点关系的精简版dump
        export DUMP_GRAPH_PATH=/home/dumpgraph  # 通过该环境变量指定dump图文件存储路径
        ```

        训练任务启动后，会在\$\{DUMP_GRAPH_PATH\}/pid_\$\{pid\}_deviceid_\$\{deviceid\}路径下生成若干dump图文件，包括后缀为“.pbtxt”和“.txt”的dump文件。由于dump的数据文件较多且文件都较大，若非问题定位需要，可以不生成dump图。

    - 若开发者期望程序编译运行过程中产生的文件落盘到统一存储目录，可通过环境变量ASCEND_CACHE_PATH与ASCEND_WORK_PATH分别设置共享文件的存储路径与进程独享文件的存储路径。

        ```bash
        export ASCEND_CACHE_PATH=/repo/task001/cache
        export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
        ```

2. 执行脚本，启动训练进程。

    ```bash
    # 启动节点0上的训练进程
    bash tf_start_16p.sh --server_index=0 --devices_num=8
    # 启动节点1上的训练进程
    bash tf_start_16p.sh --server_index=1 --devices_num=8
    ```

### 结果说明

分布式训练执行完成后，开发者可以参考本章节检查执行结果、定位问题。

1. 检查运行结果。

    不同的训练脚本打印结果不同，若执行分布式训练的每个Device出现类似如下打印信息，说明训练任务已经正常结束。

    ![](../figures/distributed_train_1.png)

    当启用环境变量DUMP_GE_GRAPH时，会生成GE的dump图文件。

    ```bash
    export DUMP_GE_GRAPH=2
    ```

    在dump下来的图文件目录下，搜索到包含了HcomBroadcast和HcomAllReduce算子，这表明正常插入了NPU间通信的HCCL算子。

    GE的dump图示例如下：

    ![](../figures/ge_dump.png "GE的dump图")

2. 如果运行失败，和单Device训练一样，通过日志分析并定位问题。

    在$HOME/ascend/log/run/plog下查看Host侧日志plog_\*.log，$HOME为Host侧用户根目录。

    在单Device执行成功，多Device执行失败的情况下，一般为集合通信的问题，如下图所示。可以参见《[HCCL集合通信库用户指南](https://hiascend.com/document/redirect/CannCommunityHcclUg)》的“故障诊断”章节进行问题处理。

    ![](../figures/hccl_faq.png "集合通信问题")

如果运行失败，通过日志分析并定位问题。

在Host侧运行应用程序产生的运行日志路径：\$HOME/ascend/log/run/plog/plog-pid_\*.log。

在Device侧运行应用程序产生的运行日志路径：\$HOME/ascend/log/run/device-id/device-pid_\*.log。

$HOME为Host侧用户根目录。

> [!NOTE]说明
> 了解更多介绍，请参考《[日志参考](https://hiascend.com/document/redirect/CannCommunitylogref)》。

一般通过ERROR级别的日志，识别问题产生模块，根据具体日志内容判定问题产生原因。

错误日志样例：

![](../figures/error_log.png "错误日志样例")

问题定位思路如下表所示：

| ModuleName | 出错流程 | 解决思路 |
|------------|----------|----------|
| 系统类报错 | 环境与版本配套错误 | 系统类报错，优先排查版本配套是否正确，系统是否正常安装。 |
| GE | GE图编译或校验问题 | 校验类报错，通常会给出明确的错误原因，此时需要针对性地修改网络脚本，以满足相关要求。 |
| Runtime | 环境异常导致初始化问题或图执行问题 | 对于初始化异常，优先排查当前运行环境配置是否正确，当前环境是否有其他进程占用。 |
