# 自动迁移

## 了解自动迁移工具

- 功能介绍

  TF Adapter提供了TensorFlow 2.6.5网络迁移工具，该工具适用于原生的TensorFlow训练脚本迁移场景，AI算法工程师通过该工具分析原生的TensorFlow Python API在AI处理器上的支持度情况，同时将原生的TensorFlow训练脚本自动迁移成AI处理器支持的脚本，迁移后的脚本能在AI处理器上执行训练，功能跑通。对于无法自动迁移的API，您可以参考工具输出的迁移报告，对训练脚本进行相应的适配修改。

- 获取路径

  CANN软件安装完成后，迁移工具在“$\{TFPLUGIN_INSTALL_PATH\}/npu_device/convert_tf2npu/”目录下。

- 使用限制

  在使用工具进行模型迁移前，先来了解对原始训练脚本的限制：

  1. 要求原始脚本在GPU/CPU上运行成功，精度收敛。
  2. 要求原始脚本仅使用[TensorFlow 2.6官方API](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf)，或者当脚本中以tf.compat.v1形式调用TensorFlow 1.x API时，支持使用Horovod API。

     若用户脚本使用了其他第三方API，当前工具暂不支持迁移。例如：

     1. 不支持原生Keras API，但支持tf.keras的相关API。
     2. 不支持CuPy API，即便原始脚本能在GPU上运行成功，但不能保证在AI处理器运行成功。

  3. 原始脚本中的TensorFlow模块需要按照如下方式添加引用，否则工具迁移后，无法生成准确的迁移报告（但不影响脚本迁移功能）。

     ```bash
     import tensorflow as tf
     import horovod.tensorflow as hvd
     ```

  4. 当前版本不支持float64/complex64/complex128/DT_VARIANT数据类型。
  5. 关于分布式脚本迁移的限制：
     1. 使用工具迁移前，需要手工添加数据集分片操作，具体请参考[分布式训练脚本适配（兼容单卡）](manual_porting.md#分布式训练脚本适配兼容单卡)中的“不同worker上的数据集分片”。
     2. 当前工具仅支持对使用了TensorFlow Keras优化器（包括SGD/RMSprop/Adam/Ftrl/Adagrad/Adadelta/Adamax/Nadam）的分布式脚本进行自动迁移，其他分布式脚本需要参考[分布式训练脚本适配（兼容单卡）](manual_porting.md#分布式训练脚本适配兼容单卡)进行手工迁移。
     3. 如果用户原始脚本中使用了LossScaleOptimizer，当前工具仅支持将tf.keras.mixed_precision.LossScaleOptimizer迁移为[npu.train.optimizer.NpuLossScaleOptimizer](../../apiref/npu-train-optimizer-npulossscaleoptimizer.md)，对于其他类型的LossScaleOptimizer，您应当先切换为tf.keras.mixed_precision.LossScaleOptimizer，进行功能精度验证后再手工替换为[npu.train.optimizer.NpuLossScaleOptimizer](../../apiref/npu-train-optimizer-npulossscaleoptimizer.md)。

  6. 迁移工具目前无法自动使能循环下沉功能，如果原始脚本中使用了循环下沉，则需要用户手工使能NPU的循环下沉能力，具体请参考[训练循环下沉时设置NPU上的循环次数](./manual_porting.md#训练循环下沉时设置npu上的循环次数)。

## 前提条件

在AI处理器进行模型迁移之前，建议用户事先准备好基于TensorFlow 2.6.5开发的训练模型以及配套的数据集，并要求在GPU或CPU上正常执行，精度收敛，且达到预期精度和性能要求。同时记录相关精度和性能指标，用于后续在AI处理器进行精度和性能对比。

## 迁移操作步骤

1. 安装依赖。

    ```bash
    pip3 install pandas==1.3.5
    pip3 install openpyxl
    pip3 install google_pasta
    ```

2. 训练脚本扫描和自动迁移。

    进入迁移工具所在目录“$\{TFPLUGIN_INSTALL_PATH\}/npu_device/convert_tf2npu/”，执行命令可同时完成脚本扫描和自动迁移，例如：

    ```bash
    python3 main.py -i /root/models/examples/test -m /root/models/example/test/test.py
    ```

    其中main.py为工具入口脚本，参数说明如下所示：

    | 参数名 | 参数说明 | 可选/必选 |
    | --- | --- | --- |
    | -i | 被迁移的原始脚本路径，当前该路径仅支持配置为文件夹，不支持单个文件。<br>  - 工具仅对-i参数指定的文件夹下的.py文件进行扫描和迁移。<br>  - 如果用户原始脚本跨目录存放，则建议放到一个目录执行迁移命令，或者在对应目录下依次执行迁移命令。 | 必选 |
    | -o | 指定迁移后的脚本路径，该路径不能为原始脚本路径的子目录。<br>该参数可选，如果不指定，默认生成在当前路径下，例如output_npu_20220517172706/xxx_npu_20220517172706。 | 可选 |
    | -r | 指定生成的迁移报告路径，该路径不能为原始脚本路径的子目录。<br>该参数可选，如果不指定，默认生成在当前路径下，例如report_npu_20220517172706。 | 可选 |
    | -m | Python执行入口文件。<br>如果原始脚本中没有main函数，由于迁移工具无法识别入口函数，因此无法进行NPU资源初始化，以及NPU训练相关配置。<br>对于以上场景，需要通过-m参数指定Python执行的入口文件，以便工具可以将用户脚本进行彻底迁移，保证后续训练的正常执行。<br>配置示例：-m /root/models/xxx.py | 可选 |
    | -d | 如果原始脚本支持分布式训练，迁移时需要指定原始脚本使用的分布式策略，便于工具对分布式脚本进行自动迁移。取值：<br>  - tf_strategy：表示原始脚本使用tf.distribute.Strategy分布式策略。<br>  - horovod：表示原始脚本使用horovod分布式模块。 | 可选 |
    | -c | 如果在脚本使用了tf.compat.v1 API，控制以TensorFlow 1.x行为执行，需要在执行脚本转换命令时添加-c或者--compat。 | 可选 |

    > [!NOTE]说明
    > 通过python3 main.py -h可以获取迁移工具使用帮助。

    - 迁移过程中，打印如下信息，表明正在扫描相关文件进行脚本迁移。

      ![迁移过程信息](../figures/migration_process_info.png)

    - 迁移结束后，生成迁移后的脚本，以及迁移报告。

      ![迁移结束信息](../figures/migration_end_info.png)

      - 如果没有生成failed_report.txt，一般迁移后的模型即可直接在AI处理器执行训练，用户可尝试执行训练，如果训练失败，可详细分析迁移报告，同时酌情修改训练脚本再次训练，如果仍然训练失败，请在本源码仓提issue。
      - 如果生成了failed_report.txt，请优先根据报错修改训练脚本，再次执行训练。

## 迁移报告说明

- success_report.txt：记录工具对脚本的全部修改点，例如：

    ```text
    # 表示adain.py第3行新增头文件引用
    /root/models/examples/adain/adain.py:3 import npu_device as npu
    # 表示adain.py第4行新增npu虚拟设备初始化
    /root/models/examples/adain/adain.py:4 npu.open().as_default()
    ```

- failed_report.txt：记录迁移过程中的报错信息以及不支持的api，例如：

    ```text
    # 表示test.py第4行使用的API在NPU上不支持，且无迁移方案，需用户更换为支持的API
    /root/models/examples/test/test.py:4, NPU Unsupported API: tf.compat.v1.enable_eager_execution. Please replace it with an NPU supported API.
    # 表示test.py第5行使用了第三方非TF官网API，需用户手工修改脚本
    /root/models/examples/test/test.py:5, NPU Unsupported API: cupy.array. Please modify the user script manually.
    ```

- api_analysis_report.xlsx：API支持度分析报告，用户可根据修改建议修改训练脚本。
- need_migration_doc.txt：需要用户手工迁移的API。
- api_brief_report.txt：汇总脚本中API支持度统计结果，例如：

    ```text
    # 未去重的统计结果，分类和API支持度表中的一致
     1.In brief: Total API: 231, in which Support: 222, API support after migration: 3, Network training support after migration: 2, Not support but no impact on migration: 1, Not support or recommended: 2, Compatible: 1, Deprecated: 0, Analysing: 0
     # 去重后的统计结果，分类和API支持度表中的一致
     2.After eliminate duplicate: Total API: 98, in which Support: 92, API support after migration: 2, Network training support after migration: 1, Not support but no impact on migration: 1, Not support or recommended: 1, Compatible: 1, Deprecated: 0, Analysing: 0
    ```
