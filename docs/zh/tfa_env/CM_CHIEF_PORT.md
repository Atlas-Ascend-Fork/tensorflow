# CM_CHIEF_PORT

## 功能描述

TensorFlow分布式训练或推理场景下，用户可以选择不使用rank table文件，通过组合使用环境变量[CM_CHIEF_IP](CM_CHIEF_IP.md)、[CM_CHIEF_PORT](CM_CHIEF_PORT.md)、[CM_CHIEF_DEVICE](CM_CHIEF_DEVICE.md)、[CM_WORKER_SIZE](CM_WORKER_SIZE.md)、[CM_WORKER_IP](CM_WORKER_IP.md)的方式自动生成资源信息，完成集合通信组件初始化。

本环境变量“CM_CHIEF_PORT”用于配置Master节点的监听端口。

该环境变量需要配置为整数，取值范围“0~65520”，请确保端口未被其他进程占用。

## 配置示例

```bash
export CM_CHIEF_PORT=6000
```

## 使用约束

此环境变量不能与[RANK_TABLE_FILE](RANK_TABLE_FILE.md)、[RANK_ID](RANK_ID.md)、[RANK_SIZE](RANK_SIZE.md)混合使用。

## 支持的型号

Atlas A2系列产品

Atlas训练系列产品
