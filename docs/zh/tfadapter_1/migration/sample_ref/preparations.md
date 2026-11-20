# 训练前准备

## 原始模型获取

ResNet50为一个深度残差网络，可用于对CIFAR-10和ImageNet的1000类数据集进行分类。

单击[链接](https://github.com/tensorflow/models/tree/r2.1.0/official)获取ResNet原始网络脚本。

主要文件目录结构如下所示（只列出部分涉及文件，更多文件请查看获取的ResNet原始网络脚本）：

```text
├── r1   // 原始模型目录
│   ├── resnet    // ResNet主目录
│        ├── __init__.py
│        ├── imagenet_main.py      // 基于ImageNet数据集训练网络模型
│        ├── imagenet_preprocessing.py     // ImageNet数据集数据预处理模块
│        ├── resnet_model.py     // ResNet模型文件
│        ├── resnet_run_loop.py    // 数据输入处理与运行循环（训练、验证、测试）
│        ├── README.md   // 项目介绍文件
│   ├── utils
│        ├── export.py     // 数据接收函数，定义了导出的模型将会对何种格式的参数予以响应
├── utils
│   ├── flags
│        ├── core.py         // 包含了参数定义的公共接口
│   ├── logs
│        ├── hooks_helper.py     //自定义创建模型在测试/训练时的工具，例如每秒钟计算步数的功能、每N步或捕获CPU/GPU分析信息的功能等
│        ├── logger.py      // 日志工具
│   ├── misc
│        ├── distribution_utils.py       // 进行分布式运行模型的辅助函数
│        ├── model_helpers.py      // 定义了一些能被模型调用的函数，例如控制模型是否停止
```

## 数据集准备

1. 准备数据集
    1. 获取数据集

        本训练示例以ImageNet2012数据集为例，从ImageNet官方网站[https://www.image-net.org/](https://www.image-net.org/)获取数据集。将准备好的数据集压缩包上传到训练环境上（需确认数据集是否完整）。

    2. 数据集目录参考（本示例在“ /data/dataset/“路径下）：

        ```text
        ├──imagenet2012
        │   ├──ILSVRC2012_img_train.tar
        │   ├──ILSVRC2012_img_val.tar
        │   ├──ILSVRC2012_bbox_train_v2.tar.gz
        ```

    3. 执行如下命令，创建并运行脚本文件，创建“train“、“val“、“bbox“和“imagenet_tf“目录，分别解压train、val、bbox数据集压缩包到对应的目录。
        1. 执行如下命令，创建并打开“prepare_dataset.sh“文件。

            **vim prepare_dataset.sh**

        2. 在文件中添加如下脚本命令。

            ```bash
            #!/bin/bash
            mkdir -p train val bbox imagenet_tf
            tar -xvf ILSVRC2012_img_train.tar -C train/
            tar -xvf ILSVRC2012_img_val.tar -C val/
            tar -xvf ILSVRC2012_bbox_train_v2.tar.gz -C bbox/
            ```

        3. 执行**:wq!**命令保存文件并退出。
        4. 执行如下命令运行脚本文件。

            **bash prepare_dataset.sh**

            如果解压后的“train”目录下仍为.tar文件，请在“train”目录下执行以下命令解压。

            `find . -name "*.tar" | while read LINE ; do mkdir -p "${LINE%.tar}"; tar -xvf "${LINE}" -C "${LINE%.tar}"; rm -f "${LINE}"; done`

    4. 检查数据集目录（本示例在“ /data/dataset/“路径下）。

        ```text
        ├──imagenet2012
        │   ├──ILSVRC2012_img_train.tar
        │   ├──ILSVRC2012_img_val.tar
        │   ├──ILSVRC2012_bbox_train_v2.tar.gz
        │   ├──bbox/
        │   ├──train/
        │   ├──val/
        ```

2. 转换数据集为TFRecord格式。
    1. 下载源代码。

        **git clone** [https://github.com/tensorflow/models.git](https://github.com/tensorflow/models.git)

    2. 执行如下命令进入源代码的“datasets“目录，并预处理验证数据。

       ```bash
       cd models-master/research/slim/datasets/

       python preprocess_imagenet_validation_data.py /data/dataset/imagenet2012/val/ imagenet_2012_validation_synset_labels.txt  # 预处理验证数据
       ```

    3. 将标注的XML文件转换为单个CSV文件。

       ```bash
       python process_bounding_boxes.py /data/dataset/imagenet2012/bbox/ imagenet_lsvrc_2015_synsets.txt | sort > imagenet_2012_bounding_boxes.csv
       ```

    4. 转换ImageNet数据集为TFRecord格式。

       ```bash
       python build_imagenet_data.py --output_directory=/data/dataset/imagenet2012/imagenet_tf --train_directory=/data/dataset/imagenet2012/train --validation_directory=/data/dataset/imagenet2012/val
       ```

3. 查看转换好的数据集。

    ```text
    ├─ imagenet2012
    ├─├─imagenet_tf
    │     ├──train-00000-of-01024
    │     ├──train-00001-of-01024
    │     ├──train-00002-of-01024
    │     ...
    │     ├──validation-00000-of-00128
    │     ├──validation-00001-of-00128
    │     ├──validation-00002-of-00128
    │     ...
    ```
