# 快速入门

本节通过一个简单的TensorFlow脚本迁移样例，帮助用户快速了解TensorFlow脚本迁移到昇腾平台并执行的方法。

1. 脚本迁移。
    - 迁移前脚本示例：

        ```python
        import tensorflow as tf
        a = tf.random_normal([2, 3], dtype=tf.float32)
        b = tf.random_uniform([2, 3], dtype=tf.float32)
        c = tf.add(a, b)
        with tf.Session() as sess:
            result = sess.run(c)
            print("result: ", result)
        ```

    - 迁移后脚本示例：

        ```python
        import tensorflow as tf
        # 导入NPU相关库
        from npu_bridge.npu_init import *
        a = tf.random_normal([2, 3], dtype=tf.float32)
        b = tf.random_uniform([2, 3], dtype=tf.float32)
        c = tf.add(a, b)
        # 增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
        config = tf.ConfigProto(allow_soft_placement=True)
        # 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 需要显示关闭TensorFlow的remapping、memory_optimization功能，避免与NPU中的功能冲突。
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 显式关闭
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 显式关闭
        with tf.Session(config=config) as sess:
            result = sess.run(c)
            print("result: ", result)
        ```

2. 脚本执行。
    1. 配置环境变量。

        ```bash
        # 配置CANN软件环境变量，以root用户默认安装路径为例
        source /usr/local/Ascend/cann/set_env.sh

        # TF Adapter python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
        export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

        export JOB_ID=10087        # 训练任务ID，用户自定义，仅支持大小写字母，数字，中划线，下划线。不建议使用以0开头的纯数字
        export ASCEND_DEVICE_ID=0  # 指定AI处理器的逻辑ID，单卡训练也可不配置，默认为0，在0卡执行训练
        ```

    2. 执行脚本。

        假设迁移后的脚本命名为tf_quickstart.py，命令示例如下：

        ```bash
        python3 tf_quickstart.py
        ```
