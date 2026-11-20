# Quick Start

This section uses a simple example to help you quickly understand how to port TensorFlow scripts to the Ascend platform and execute them.

1. Port the script.
    - Script before porting:

        ```python
        import tensorflow as tf
        a = tf.random_normal([2, 3], dtype=tf.float32)
        b = tf.random_uniform([2, 3], dtype=tf.float32)
        c = tf.add(a, b)
        with tf.Session() as sess:
            result = sess.run(c)
            print("result: ", result)
        ```

    - Script after porting:

        ```python
        import tensorflow as tf
        # Import NPU-related libraries.
        from npu_bridge.npu_init import *
        a = tf.random_normal([2, 3], dtype=tf.float32)
        b = tf.random_uniform([2, 3], dtype=tf.float32)
        c = tf.add(a, b)
        # Add allow_soft_placement=True for the session configurations to allow TensorFlow to automatically allocate devices.
        config = tf.ConfigProto(allow_soft_placement=True)
        # Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # Explicitly disable the remapping and memory_optimization functions of TensorFlow to avoid conflicts with the functions of the NPU.
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Explicitly disable the function
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # Explicitly disable the function.
        with tf.Session(config=config) as sess:
            result = sess.run(c)
            print("result: ", result)
        ```

2. Execute the script.
    1. Set environment variables.

        ```bash
        # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
        source /usr/local/Ascend/cann/set_env.sh

        # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
        export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

        export JOB_ID=10087        # User-defined training job ID. Only letters, digits, hyphens (-), and underscores (_) are supported. You are advised not to use a number starting with 0.
        export ASCEND_DEVICE_ID=0  # Logical ID of the AI processor, optional in single-device training and defaulted to 0, indicating that training is performed on device 0.
        ```

    2. Execute the script.

        Assume that the ported script is named  **tf_quickstart.py**. The command example is as follows:

        ```bash
        python3 tf_quickstart.py
        ```
