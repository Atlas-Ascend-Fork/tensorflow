# 自动迁移

描述如何使用迁移工具将TensorFlow 1.15网络自动迁移到昇腾平台。

## 了解自动迁移工具

- 【功能介绍】

    Ascend平台提供了TensorFlow 1.15网络迁移工具，该工具适用于原生的TensorFlow训练脚本迁移场景，AI算法工程师可通过该工具分析原生的TensorFlow Python API和Horovod Python API在AI处理器上的支持度情况，同时将原生的TensorFlow训练脚本自动迁移成AI处理器支持的脚本。对于无法自动迁移的API，您可以参考工具输出的迁移报告，对训练脚本进行相应的适配修改。

- 【获取路径】
  - CANN软件安装完成后，迁移工具在“\$\{TFPLUGIN_INSTALL_PATH\}/npu_bridge/convert_tf2npu/”目录下，其中\$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包的安装路径。
  - 您也可以从[Gitcode仓](https://gitcode.com/cann/tensorflow)获取“convert_tf2npu”文件夹，并将“convert_tf2npu”文件夹上传到Linux或Windows环境上任意目录即可。

- 【使用约束】在使用工具进行模型迁移前，先来了解对原始训练脚本的约束：
    1. 要求原始脚本在GPU/CPU上成功执行，精度收敛。
    2. 要求原始脚本仅使用[TensorFlow 1.15官方API](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)和[Horovod官方API](https://horovod.readthedocs.io/en/stable/api.html#module-horovod.tensorflow)，若用户脚本使用了其他第三方API，当前工具暂不支持迁移。例如：
        - 不支持原生Keras API，但支持tf.keras的相关API。
        - 不支持CuPy API，即便原始脚本能在GPU上运行成功，但不能保证在AI处理器运行成功。

    3. 原始脚本中的TensorFlow模块和Horovod模块最好按照如下方式引用，否则工具迁移后，无法生成准确的迁移报告（但并不影响脚本迁移）。

        ```python
        import tensorflow as tf
        import tensorflow.compat.v1 as tf
        import horovod.tensorflow as hvd
        ```

    4. 当前不支持tf.keras和原生Keras的Loss Scale功能迁移。
    5. 其他约束请参见[系统约束与限制](../../introduction.md#系统约束与限制)。

## 前提条件

在AI处理器进行模型迁移之前，建议用户事先准备好基于TensorFlow 1.15开发的训练模型以及配套的数据集，并要求在GPU或CPU上跑通，精度收敛，且达到预期精度和性能要求。同时记录相关精度和性能指标，用于后续在AI处理器进行精度和性能对比。

## 迁移操作步骤

1. 安装依赖。

    ```bash
    pip3 install pandas==1.3.5
    pip3 install xlrd==1.2.0
    pip3 install openpyxl
    pip3 install tkintertable
    pip3 install google_pasta
    ```

2. 训练脚本扫描和自动迁移。

    该工具支持在Linux或Windows环境进行脚本迁移。

    - Linux环境操作步骤：

      进入迁移工具所在目录“\$\{TFPLUGIN_INSTALL_PATH\}/npu_bridge/convert_tf2npu/”，其中$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包的安装路径，执行如下命令可同时完成脚本扫描和自动迁移，例如：

      ```bash
      python3 main.py -i /root/models/official/resnet
      ```

      其中main.py为工具入口脚本，参数说明如下所示。

      | 参数名 | 参数说明 | 可选/必选|
      |--- | --- | ---|
      |-i | 被迁移的原始脚本路径，当前该路径仅支持配置为文件夹，不支持单个文件。<br>说明：<br>- 工具仅对-i参数指定的文件夹下的.py文件进行扫描和迁移。<br>- 如果用户原始脚本跨目录存放，则建议放到同一个目录执行迁移命令，或者在对应目录下依次执行迁移命令。 | 必选|
      |-o | 指定迁移后的脚本路径，该路径不能为原始脚本路径的子目录。<br>该参数可选，如果不指定，默认生成在当前路径下，例如output_npu_20210401150929/xxx_npu_20210401150929。 | 可选  |
      |-r | 指定生成的迁移报告路径，该路径不能为原始脚本路径的子目录。<br>该参数可选，如果不指定，默认生成在当前路径下，例如report_npu_20210401150929。 | 可选|
      |-m | Python执行入口文件。<br>如果原始脚本使用了tf.keras/hvd接口，且脚本中没有main函数，由于迁移工具无法识别入口函数，因此无法进行NPU资源初始化和NPU训练相关配置。<br>对于以上场景，需要通过-m参数指定Python执行的入口文件，以便工具可以将用户脚本进行彻底迁移，保证后续训练的顺利执行。<br>配置示例：-m /root/models/xxx.py | 可选|
      |-d | 如果原始脚本支持分布式训练，需要指定原始脚本使用的分布式策略，便于工具对分布式脚本进行自动迁移。取值：<br>- tf_strategy：表示原始脚本使用tf.distribute.Strategy分布式策略<br>- horovod：表示原始脚本使用horovod分布式策略<br>目前session run分布式脚本无法彻底进行自动迁移，使用工具自动迁移完后，需要参考[自动迁移后如何进行sess.run分布式脚本改造](../common_operation/sessrun_distributed_after_auto_port.md)进行后续的手工改造。 | 分布式必选  |

       > [!NOTE]说明
       > 通过python3 main.py -h可以获取迁移工具使用帮助。

    - Windows环境操作步骤：

      ```bash
      python3 main_win.py
      ```

        在弹出的窗口根据界面提示进行操作。

3. 迁移过程中，打印如下信息，表明正在扫描相关文件进行脚本迁移。

    **图 1**  迁移过程信息
    ![](../figures/porting_process_info.png "迁移过程信息")

4. 迁移结束后，生成迁移后的脚本，以及迁移报告。

    **图 2**  迁移结束信息
    ![](../figures/porting_end_info.png "迁移结束信息")

    - 如果没有生成failed_report.txt，一般迁移后的模型即可直接在AI处理器执行训练，如果训练失败，可详细分析迁移报告，同时酌情修改训练脚本再次训练，如果仍然训练失败，请在本源码仓提issue。
    - 如果生成了failed_report.txt，请优先根据报错修改训练脚本，再执行训练。

## 迁移报告说明

- success_report.txt：记录工具对脚本的全部修改点。
- failed_report.txt：记录迁移过程中的报错信息以及不支持的API。
- api_analysis_report.xlsx：API支持度分析报告，用户可根据修改建议修改训练脚本。
- need_migration_doc.txt：需要用户手工迁移的API。
- api_brief_report.txt：汇总脚本中API支持度统计结果。

## 后续处理（可选）

Ascend平台提供了功能调试、性能/精度调优等功能，自动迁移后，可通过如下session配置的方式使能相关功能。

1. 检查迁移后的脚本是否存在“init_resource”。
    - 如果存在，则参考如下示例，在init_resource函数中传入session_config的配置，需要注意，仅[initialize_system](../../apiref/npu_ops/initialize_system.md)中支持的配置项可在init_resource函数的config中进行配置，若需配置其他功能，请在运行配置中添加，可参见步骤2。

        ```python
        if __name__ == '__main__':
          # 增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
          session_config = tf.ConfigProto(allow_soft_placement=True)
          # 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # 配置session参数
          custom_op.parameter_map["profiling_mode"].b = True
          ... ...

          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

    - 如果不存在，则直接执行下一步。

2. 在运行配置中添加相关session配置。
     - 针对Estimator模型的脚本，在迁移后的脚本中查找“npu_run_config_init”，找到运行配置函数，例如示例中的“run_config”，在运行配置参数中添加相关session参数，如下面示例中的“aoe_mode”参数。

         ```python
         session_config = tf.ConfigProto(allow_soft_placement=True)
         # 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
         custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
         custom_op.name = 'NpuOptimizer'
         # 配置session参数
         custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")

         run_config = tf.estimator.RunConfig(
          train_distribute=distribution_strategy,
          session_config=session_config,
          save_checkpoints_secs=60*60*24)

         classifier = tf.estimator.Estimator(
          model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
         ```

     - 针对sess.run模式的脚本，在迁移后的脚本中查找“npu_config_proto”，找到运行配置参数（例如下面示例中的“session_config”），在运行配置参数中添加相关session参数，如下面示例中的“aoe_mode”参数。

         ```python
         session_config = tf.ConfigProto(allow_soft_placement=True)
         # 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
         custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
         custom_op.name = 'NpuOptimizer'
         # 配置session参数
         custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
         config = npu_config_proto(config_proto=session_config)
         with tf.Session(config=config) as sess:
           sess.run(tf.global_variables_initializer())
           sess.run(interaction_table.init)
         ```

     - 针对Keras模式的脚本，在迁移后的脚本中查找“set_keras_session_npu_config”函数，找到运行配置参数（例如下面示例中的“config_proto”），在运行配置参数中添加相关session参数，如下面示例中的“aoe_mode”参数。

         ```python
         import tensorflow as tf
         import tensorflow.python.keras as keras
         from tensorflow.python.keras import backend as K
         from npu_bridge.npu_init import *

         config_proto = tf.ConfigProto(allow_soft_placement=True)
         # 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
         custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
         custom_op.name = 'NpuOptimizer'
         # 配置session参数
         custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
         npu_keras_sess = set_keras_session_npu_config(config=config_proto)

         # 数据预处理...
         # 模型搭建...
         # 模型编译...
         # 模型训练...
         ```
