# 精度分析工具部署

## 工具简介

为方便开发者进行训练场景的精度调试，一键式精度分析工具提供了对训练网络进行精度分析的常用功能，主要包括：

- [浮点异常检测](floating-point_exception_detection.md)
- [融合异常检测](fusion_exception_detection.md)
- [整网数据比对](network_accuracy_comparison.md)
- [随机错误检测](random_error_detection.md)

该工具封装了TF Adapter的运行参数，帮助用户更方便地使能相关业务特性，同时对CANN软件包中的精度分析工具“**msaccucmp.py**”进行了**封装和功能扩展**，从而帮助开发人员快速分析精度问题。

## 使用约束

- 不能同时采集溢出数据和精度数据。
- 采集溢出数据或精度数据时，都可能会产生较多结果文件，导致磁盘空间不足，请适当控制迭代次数。

## 工具部署

从[https://gitee.com/ascend/tools](https://gitee.com/ascend/tools)下载**precision_tool**文件夹，上传到训练工作目录下，无需安装。

目录结构示例如下：

```text
├── resnet                              // 训练工作目录
│    ├── __init__.py
│    ├── imagenet_main.py              // 基于ImageNet数据集训练网络模型
│    ├── imagenet_preprocessing.py     // ImageNet数据集数据预处理模块
│    ├── resnet_model.py               // resnet模型文件
│    ├── resnet_run_loop.py            // 数据输入处理与运行循环（训练、验证、测试）
│    ├── cifar10_main                  // 训练执行入口文件
│    ├── ...
│    ├── precision_tool           // 一键式精度分析工具目录
│    │    ├── cli.py
│    │    ├── ...
```

- 如果CANN开发/运行环境合一部署，只需将**precision_tool**文件夹，上传到训练工作目录下即可。
- 如果CANN开发/运行环境独立部署，需要将**precision_tool**文件夹上传到CANN运行环境的训练工作目录下，同时将**precision_tool**上传到CANN开发环境的任意目录下。

    > [!NOTE]说明
    >
    > - CANN运行环境（包含AI处理器，即启动NPU训练的环境），在训练精度调试工作中，主要用于训练时精度数据的采集。
    > - CANN开发环境，在训练精度调试工作中，主要用于精度数据分析。

## 典型使用流程

CANN开发环境与运行环境合一部署的场景下，使用一键式精度分析工具precision_tool进行精度分析的总体流程如下图所示。

**图 1**  CANN开发/运行环境合一部署
![](../figures/dev_run_merge.png "CANN开发-运行环境合一部署")

CANN开发环境与运行环境独立部署的场景下，使用一键式精度分析工具precision_tool进行精度分析的总体流程如下图所示。

**图 2**  CANN开发/运行环境独立部署
![](../figures/dev_run_part.png "CANN开发-运行环境独立部署")
