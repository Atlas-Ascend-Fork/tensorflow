# Automated Porting

This section describes how to use the porting tool to automatically port the TensorFlow 1.15 network to the Ascend platform.

## About the Porting Tool

- **Function Overview**

    The Ascend platform provides a porting tool targeting at TensorFlow 1.15. AI algorithm engineers can use the tool to analyze the support for TensorFlow and Horovod Python APIs on the  AI processor, and automatically port native TensorFlow training scripts to those supported by the  AI processor. For APIs unportable by the tool, modify your training scripts according to the tool report.

- **How to Obtain**
  - After the CANN software is installed, the porting tool is stored in the  **$\{TFPLUGIN_INSTALL_PATH\}/npu_bridge/convert_tf2npu/**  directory. $\{TFPLUGIN_INSTALL_PATH\} indicates the installation path of the TF Adapter package.
  - You can also obtain the "convert_tf2npu" folder from the [Gitcode repository](https://gitcode.com/cann/tensorflow) and upload the "convert_tf2npu" folder to any directory in a Linux or Windows environment.

- \[Restrictions\] You should consider the following restrictions on your original training script before using the tool.
    1. The original script can run on the GPU or CPU for accuracy convergence.
    2. The original script must be developed using  [official TensorFlow 1.15 APIs](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)  and  [official Horovod APIs](https://horovod.readthedocs.io/en/stable/api.html#module-horovod.tensorflow). Otherwise, the porting tool cannot port the script. You can refer to the following cases.
        1. Native Keras APIs are not supported. However, tf.keras APIs are supported.
        2. CuPy APIs are not supported. It does not promise a successful execution on the  AI processor  even if the original script can run properly on the GPU.

    3. It is recommended that the TensorFlow and Horovod modules in the original script be referenced as follows. Otherwise, an accurate porting report cannot be generated. \(This does not affect the script porting.\)

        ```python
        import tensorflow as tf
        import tensorflow.compat.v1 as tf
        import horovod.tensorflow as hvd
        ```

    4. Currently, the loss scaling function of  **tf.keras**  and native  **Keras**  APIs is not supported after porting.
    5. For details about other restrictions, see  [Restrictions](../../introduction.md#restrictions).

## Prerequisites

Before model porting to the  AI processor, prepare a training model developed on TensorFlow 1.15 and a matched dataset, and run the model on the GPU or CPU to test if the accuracy is converged as expected. In addition, record the accuracy and performance specifications for comparison on the  AI processor  later on.

## Procedure

1. Install dependencies.

    ```bash
    pip3 install pandas==1.3.5
    pip3 install xlrd==1.2.0
    pip3 install openpyxl
    pip3 install tkintertable
    pip3 install google_pasta
    ```

2. Perform script scanning and automated porting.

    This tool supports script porting in the Linux or Windows environment.

    - The following applies to the Linux environment:

        Go to the directory  **$\{TFPLUGIN_INSTALL_PATH\}/npu_bridge/convert_tf2npu/**  where the porting tool is located.  $\{TFPLUGIN_INSTALL_PATH\}  is the installation path of the TF Adapter package. Run the following command to scan the script and perform automated porting:

        ```bash
        python3 main.py -i /root/models/official/resnet
        ```

        **main.py**  is the entry script of the tool. The following table describes the options.

        | Option | Description | Required |
        | --- | --- | --- |
        | -i | Path of the training script to be ported, which must be a folder path.<br>Notes:<br><br>  - The tool scans and ports only the .py files in the folder specified by the -i option.<br>  - If the original scripts are stored in different directories, you are advised to arrange them in the same directory or run the porting commands in sequence in each directory. | Yes |
        | -o | Path of the ported script. The path cannot be a subdirectory of the original script path.<br>Optional. If it is not specified, the current path is used by default, for example, output_npu_20210401150929/xxx_npu_20210401150929. | No |
        | -r | Path of the porting report. The path cannot be a subdirectory of the original script path.<br>Optional. If it is not specified, the current path is used by default, for example, report_npu_20210401150929. | No |
        | -m | Python execution entry point file.<br>If the tf.keras/hvd API is used and the script does not contain the main function, NPU resource initialization and NPU training configuration cannot be performed as the porting tool cannot identify the entry function.<br>In that case, you need to use -m to specify the entry point file for Python execution, so that the tool can completely port the user script for subsequent training.<br>Example: -m /root/models/xxx.py | No |
        | -d | If the original script supports distributed training, you need to specify the distribution policy used by the original script so that the tool can automatically port the distributed script. Values:<br><br>  - tf_strategy: The original script uses the tf.distribute.Strategy distribution policy.<br>  - horovod: The original script uses the Horovod distribution policy.<br><br>Currently, sess.run distributed scripts cannot be automatically ported. After using the tool for automated porting, manual modifications are required based on [How Do I Reconstruct the Sess.run Distributed Script After Automated Porting?](../common_operation/sessrun_distributed_after_auto_port.md). | Yes for distributed training |

        > [!NOTE]NOTE
        >**python3 main.py -h**: displays the help information of the porting tool.

    - The following applies to the Windows environment:

        ```bash
        python3 main_win.py
        ```

        Perform operations as prompted.

3. During the porting, check for the following information, which indicates related files are being scanned for script porting.

    **Figure  1**  Porting information  
    ![](figures/porting_process_info.png "porting-information")

4. After the porting is complete, check the resultant script and porting report.

    **Figure  2**  Porting completion information  
    ![](figures/porting_end_info.png "porting-completion-information")

    - If no  **failed_report.txt**  file is generated, you can proceed with training the migrated model directly on  AI processors. In case the training fails, carefully analyze the migration report and make necessary modifications to the training script. If the issue persists, please raise an issue in this source code repository..
    - If a  **failed_report.txt**  file is generated, modify the training script based on the error message and perform the training.

## \(Optional\) Follow-up Procedure

The Ascend platform provides functions such as function debugging and performance/accuracy tuning. After automated porting, you can enable related functions by configuring the following sessions.

1. Check whether  **init_resource**  exists in the ported script.
    - If it exists, refer to the following example to pass  **session_config**  to the  **init_resource**  function. Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resource**  function. To configure other functions, add them to the run configuration. For details, see step2.

        ```python
        if __name__ == '__main__':
          # Add allow_soft_placement=True for the session configurations to allow TensorFlow to automatically allocate devices.
          session_config = tf.ConfigProto(allow_soft_placement=True)
          # Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # Configure session parameters.
          custom_op.parameter_map["profiling_mode"].b = True
          ... ...
        
          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

    - If it does not exist, go to the next step.

2. Add related session configuration to the run configuration.
    - For  **Estimator**  scripts, search for  **npu_run_config_init**  in the ported script, find the run configuration function \(such as  **run_config**  in the example\), and add related session parameters to the run configuration, such as the  **aoe_mode**  parameter in the following example:

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)
        # Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        # Configure session parameters.
        custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
        
        run_config = tf.estimator.RunConfig(
          train_distribute=distribution_strategy,
          session_config=session_config,
          save_checkpoints_secs=60*60*24)
        
        classifier = tf.estimator.Estimator(
          model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
        ```

    - For  **sess.run**  scripts, search for  **npu_config_proto**  in the ported script, find the run configuration function \(such as  **session_config**  in the example\), and add related session parameters to the run configuration, such as the  **aoe_mode**  parameter in the following example:

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)
        # Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        # Configure session parameters.
        custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
        config = npu_config_proto(config_proto=session_config)
        with tf.Session(config=config) as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(interaction_table.init)
        ```

    - For  **Keras**  scripts, search for the  **set_keras_session_npu_config**  function in the ported script, find the run configuration function \(such as  **config_proto**  in the example\), and add related session parameters to the run configuration, such as the  **aoe_mode**  parameter in the following example:

        ```python
        import tensorflow as tf
        import tensorflow.python.keras as keras
        from tensorflow.python.keras import backend as K
        from npu_bridge.npu_init import *
        
        config_proto = tf.ConfigProto(allow_soft_placement=True)
        # Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        # Configure session parameters.
        custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
        npu_keras_sess = set_keras_session_npu_config(config=config_proto)
        
        # Preprocess data...
        # Construct a model...
        # Build the model...
        # Train the model...
        ```
