# 浮点异常检测

训练网络执行过程中，当发生频繁浮点异常问题时，可参见本节内容进行浮点异常检测。

## 使用场景

在训练网络执行过程中，可能发生频繁的浮点异常情况，此时需要通过分析溢出数据，对频繁的浮点异常问题进行定界定位。

- 对于动态Loss Scale场景，一般需要对Loss Scale值下降次数较多或者直接下降为1的情况进行浮点异常定界定位，但在训练多个step的场景下，如果只是某个step出现了溢出，则可能是正常的偶发溢出，一般在开启Loss Scale的情况下，会自动跳过该step的训练结果，梯度不更新，对于这种偶发溢出场景一般可以不用关注。
- 对于静态Loss Scale场景，一般情况下即便是溢出较少的情况，也可能需要进行浮点异常定界定位。

    > [!NOTE]说明
    > Loss Scale值的打印方法请参考[打印Loss Scale值](../performance_tuning/mixed_precision_training.md#打印loss-scale值)。

溢出数据检测的主要流程如下图所示。

**图 1**  溢出数据检测流程
![](../figures/ovf_detection_flow.png)

## 前提条件

1. 已完成[调优前检查](pre-tuning_check.md)。
2. 已完成[精度分析工具部署](accuracy_analyzer_deployment.md)。
3. 进行溢出数据分析时依赖CANN软件包中的工具，因此需要准备CANN开发环境。

## Dump溢出数据

**以下操作在NPU训练环境执行。**

1. 修改训练脚本，打开算子溢出数据采集开关。

    ```bash
    # 引用precision_tool/tf_config.py
    import precision_tool.tf_config as npu_tf_config
    
    # 1. 手工迁移网络
    # 1.1 Estimator方式
    dump_config=npu_tf_config.estimator_dump_config(action='overflow')
    npu_config = NPURunConfig(dump_config=dump_config)
    # 1.2 Session run方式
    config = npu_tf_config.session_dump_config(config, action='overflow')
    sess = tf.Session(config)
    
    # 2. 自动迁移网络
    # 若脚本中未配置custom_op，则在脚本中新增如下粗体语句
    session_config = npu_tf_config.session_dump_config(session_config, action='overflow')
    # 若脚本中已配置custom_op，如下所示，则在脚本中新增下列粗体语句更新custom_op
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op = npu_tf_config.update_custom_op(custom_op, action='overflow')
    
    # 2.1 Estimator方式
    run_config = tf.estimator.RunConfig(session_config=session_config,...)
    # 2.2 Session run方式
    with tf.Session(config=npu_config_proto(session_config)):
        ....
    # 2.3 tf.keras方式
    npu_keras_sess = set_keras_session_npu_config(config=session_config)
    ```

    > [!NOTE]说明
    >
    > - 除了此种方式，您也可以参考[溢出数据采集](../others/overflow_data_collect.md)的方法修改训练脚本，采集溢出数据，但配置较为复杂，且采集到数据之后，需要手工提取并放在相应目录下，用于后续数据分析。注意两种方式不能重复配置。
    > - 仅支持采集AI Core算子的溢出数据。

2. 执行训练，如果网络存在溢出，则在precision_data/overflow/dump下会生成溢出信息文件。

## 溢出数据分析

溢出数据分析依赖CANN软件包中的ATC工具和msaccucmp.py工具，**以下操作需要在CANN开发环境执行**。

1. 将precision_tool和precision_data文件夹上传到CANN开发环境的任意目录下，目录结构示例：

    ```txt
    ├── precision_tool              
    │    ├── cli.py                   
    │    ├── ...
    ├── precision_data              
    │    ├── overflow                   
    │    │    ├── dump
    ```

2. 安装Python第三方依赖。

    ```bash
    pip3 install rich
    ```

3. 修改工具precision_tool/lib/config目录下的config.py。

    ```bash
    # 依赖CANN软件包中的atc和msaccucmp.py工具，一般在run包安装目录，配置到父目录即可
    # 默认CANN软件包安装在/usr/local/Ascend，可以不用修改，指定目录安装则需要修改
    CMD_ROOT_PATH = '/usr/local/Ascend'
    ```

4. 启动PrecisionTool交互命令行。

    **python3 ./precision_tool/cli.py**

    进入交互命令行界面：

    **PrecisionTool \>**

    > [!NOTE]说明
    > 如需退出，可执行ctrl + c。

5. 执行**[ac -l \[limit_num\] \(-c\)](precision_tool_ommand_ref.md#ac--l-limit_num--c)**命令进行溢出数据分析。

    **PrecisionTool \> ac**

    根据数据量大小，分析过程需要时间不同，当执行过程中出现算子溢出，则会输出如下结果。

    ![](../figures/op_overflow_result.png)

    从上图可以看到：

    - 算子名为：bert_encoder_layer_10_intermediate_dense_mul_FusedMulAdd
    - 算子类型为：FusedMulAdd
    - 溢出status信息为：32，表示浮点计算有溢出。
    - 溢出类型为：AI Core算子溢出，另外还可能会有其他类型的算子溢出（例如DHA Atomic Add或L2 Atomic Add），建议用户优先考虑并解决AI Core算子溢出问题。
    - 算子的输入输出信息，包括shape、dtype、输入输出数据的最大值最小值。

    > [!NOTE]说明
    > 当出现多个算子溢出时，会出现N个溢出算子信息，默认按照算子执行顺序排序，由于后面算子溢出可能是因为前一个算子溢出导致，建议用户优先分析第一个异常算子。

6. 执行[pt \(-n\) \[\*.npy\]](precision_tool_ommand_ref.md#pt--n-npy)命令，可以查看对应dump数据块的数据信息。

    ![](../figures/fp_exception_result.png)

## 分析思路参考

进行溢出数据分析前，我们先了解下不同昇腾产品的浮点数据溢出模式：

- Atlas训练系列产品，浮点计算的溢出模式默认为“饱和模式”，且仅支持“饱和模式”。饱和模式是指当计算出现溢出时，饱和为浮点数极值（+-MAX）。
- 其他系列产品，浮点计算支持两种溢出模式：饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

进行溢出数据分析的大致思路为：

1. 查看输入输出数据值。
    - 如果输入值中没有溢出值（饱和模式：65504/NaN；INF/NaN模式：Inf/NaN），输出数据中存在溢出值，则计算存在溢出。
    - 如果输入值中存在溢出值，则需要继续分析前向算子或者用户模型的常量输入是否存在异常，否则可能在计算过程中存在溢出。

2. 查看溢出算子类型。
    - 对于自定义开发的算子，可以尝试自行进行算子溢出分析（结合算子公式和溢出值进行分析），排查自定义算子是否存在问题。
    - 对于CANN内置算子，也可以先尝试初步分析，如下为常用的分析方向：
        - 如果算子输出类型为float16，解析出的输出数据中出现类似于65504/65500（饱和模式）或者Inf/NaN（INF/NaN模式），则可以切换输出算子类型至float32计算，用户可以尝试以下两种方法：
            1. （推荐）方法一：修改混合精度模式算子黑白灰名单，调整算子精度模式，请参考[修改混合精度黑白灰名单](../performance_tuning/mixed_precision_training.md#修改混合精度黑白灰名单)。
            2. 方法二：通过keep_dtype_scope接口，指定哪些算子保持原有精度。

                ```python
                from npu_bridge.npu_init import *
                with npu_scope.keep_dtype_scope():   
                  y = tf.mul(x1,x2)
                ```

        - 如果输入输出数据中均未出现溢出值，则需要结合算子公式，分析数据计算过程中是否可能出现溢出。例如AvgPool先求和再平均，求和过程可能溢出，但平均后并未溢出。

3. 如果依旧无法解决，请在本源码仓提issue。
