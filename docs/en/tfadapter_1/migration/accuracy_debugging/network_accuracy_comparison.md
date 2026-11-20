# Network-wide Accuracy Comparison

## Overview

If the accuracy still does not meet expectations after the above steps, collect operator execution results \(dump data\) during training and compare them with results from the benchmark operator \(such as TensorFlow\). This helps quickly pinpoint operators with accuracy issues. The major steps are described as follows.

![](../figures/full_net_compare.png)

## Prerequisites

1. Floating-point exceptions have been excluded, and the overflow/underflow detection function has been disabled.
2. Fusion exceptions have been excluded, and the fusion switch has been restored to on.
3. [Model Accuracy Analyzer Deployment](accuracy_analyzer_deployment.md)  has been completed in the GPU/CPU/NPU training environment.

4. All random operations for image preprocessing have been disabled in your training script. Failure to do so will result in unavailable comparison result due to inconsistent input data. For details, see  [Disabling Random Preprocessings in the Training Script](train_script_derandomize.md).

## Dumping Benchmark Data on GPU/CPU

Use the TensorFlow debugger — tfdbg \(by adding the  **tf_debug**  code to your CPU/GPU training script\) and the precision_tool command line to generate an .npy dump file.

Perform the following operations in the GPU/CPU training environment.

1. Install Python 3 dependencies in the GPU/CPU training environment.

    ```bash
    pip3 install gnureadline pexpect
    ```

2. Edit the original training script to dump benchmark data.

    This is implemented by using the  **print_tensor\(pt\)**  command of  **tf_debug**. As the training code provides a flexible  **run\(\)**  function and there is no way to inform the script of the exact run phase where tensors should be dumped, you must edit the training code.  **Ensure that training exits immediately when a step is complete to avoid accuracy analysis bugs**.

    ```python
    # Import precision_tool/tf_config.py.
    import precision_tool.tf_config as npu_tf_config

    # If Estimator is used, add training_hooks to EstimatorSpec.
    # It is equivalent to estim_specs = tf_debug.DumpingDebugHook("precision_data/tf/tf_debug").
    estim_specs = tf.estimator.EstimatorSpec(training_hooks=[npu_tf_config.estimator_dump()])

    # If session.run is used, add the tf_debug wrapper to sess.
    # It is equivalent to sess = tf_debug.DumpingDebugWrapperSession(sess, "precision_data/tf/tf_debug").
    sess = npu_tf_config.sess_dump(sess=sess)
    ```

3. Perform GPU/CPU training.

    A number of dump directories are generated under  **precision_data/tf/tf_debug/**  based on the number of runs.

4. Use the precision_tool command line to analyze the dump files and generate the operator output tensor file.

    ```bash
    python3 precision_tool/cli.py tf_dump
    ```

5. Find the extracted tensors in the  **precision_data/tf/dump/**  directory.

    To regenerate dump data, run the following command:

    ```bash
    rm -rf precision_data/tf/dump/* && python3 precision_tool/cli.py tf_dump
    ```

## Dumping User Model on NPU

Perform the following operations in the NPU training environment. Pay attention to the following points before dumping data:

Generally, dump of the first step is enough for comparison and analysis. To avoid inaccurate comparison caused by random weights, enable checkpoints saving before training. If you find an accuracy issue with a particular step, resume the training process from the checkpoint closest to the particular step.

1. Modify  **config.py**  in the  **precision_tool/lib/config**  directory of the tool and specify the step of the data to be dumped.

    ```python
    # Set the steps to dump, for example '0|5|10'. To dump the input layer, retain the default value.
    TF_DUMP_STEP = '0'
    ```

    If  **TF_DUMP_STEP**  is not set, dump data of all iterations is collected.

2. Edit the original training script to enable dumping.

    With the following script, both dump data and dump graphs are generated.

    ```python
    # Import precision_tool/tf_config.py.
    import precision_tool.tf_config as npu_tf_config

    # 1. Manual network porting
    # 1.1 Estimator mode
    dump_config=npu_tf_config.estimator_dump_config(action='dump')
    npu_config = NPURunConfig(dump_config=dump_config)
    # 1.2 Session run mode
    config = npu_tf_config.session_dump_config(config, action='dump')
    sess = tf.Session(config)

    # 2. Automated network porting
    # If custom_op is not configured in the script, add the following statement in bold to the script:
    session_config = npu_tf_config.session_dump_config(session_config, action='dump')
    # If custom_op has been configured in the script, add the following statement in bold to the script to update custom_op:
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op = npu_tf_config.update_custom_op(custom_op, action='dump')

    # 2.1 Estimator mode
    run_config = tf.estimator.RunConfig(session_config=session_config,...)
    # 2.2 Session run mode
    with tf.Session(config=npu_config_proto(session_config)):
        ....
    # 2.3 tf.keras mode
    npu_keras_sess = set_keras_session_npu_config(config=session_config)
    ```

    > [!NOTE]NOTE
    > In addition to this method, you can also refer to  [Dump Data Collection](../others/dump_data_collect.md)  to modify the training script and collect dump data. However, the configuration is complex, and you need to manually extract the dump data and save it to the required directory for analysis. Note that the two modes are mutually exclusive.

3. Run training. The dump graph and dump data files of GE are generated in the  **precision_data/npu/debug_0**  directory.

    For details about subsequent data analysis, see  [Comparing Dump Data](#comparing-dump-data).

## Comparing Dump Data

Accuracy analysis depends on the ATC and  **msaccucmp.py**  tools in the CANN package. Perform the following operations in the CANN development environment:

1. Upload the  **precision_tool**  and  **precision_data**  directories \(containing the benchmark and NPU dump data\) to any directory in the CANN development environment. The two directories are organized as follows:

    ```text
    ├── precision_tool
    │    ├── cli.py
    │    ├── ...
    ├── precision_data
    │    ├── npu
     │    │    ├── debug_0  // NPU dump data.
    │    ├── tf
    │    │    ├── dump     // Benchmark dump data
    ```

2. Install the Python 3 dependencies.

    ```bash
    # Graphviz is optional and needs to be installed only when you need to create operator subgraphs.
    pip3 install rich graphviz
    # ubuntu/Debian
    sudo apt-get install graphviz
    # fedora/CentOS
    sudo yum install graphviz
    ```

3. Modify  **config.py**  in the  **precision_tool/lib/config**  directory.

    ```python
    # The tool depends on the atc and msaccucmp.py tools in the CANN package. Set this parameter to the CANN package installation path.
    # By default, the CANN package is installed in /usr/local/Ascend. You can retain the path or replace the path as needed.
    CMD_ROOT_PATH = '/usr/local/Ascend'
    ```

4. Start the precision_tool command line.

    **python3 ./precision_tool/cli.py**

    Enter the command line interface.

    **PrecisionTool \>**

    > [!NOTE]NOTE
    > To exit, press  **Ctrl+C**.

5. Run the  [ac -l \[limit_num\] \(-c\)](precision_tool_command_ref.md#ac--l-limit_num--c)  command for network-wide accuracy comparison.

    **PrecisionTool \> ac -c**

    The time consumption varies depending on the data size.

    The comparison result is saved in CSV format in the  **precision_data/temp/vector_compare**  directory.

    ![](../figures/fusion_exception_detect_result.png)

    You can directly inspect the CSV file. For details, see  [Network Accuracy Comparison Result File](network_accuracy_comparison_result_file.md).

6. \(Optional\) Run the  [vcs -f \[file_name\] -c \[cos_sim_threshold\] -l \[limit\]](precision_tool_command_ref.md#vcs--f-file_name--c-cos_sim_threshold--l-limit)  command to narrow down the operators with potential accuracy issues.

    By default, the  **vcs**  command returns operators with cosine similarity values less than  **0.98**. The threshold can be user-defined by using the  **-c**  argument.

    ![](../figures/precision_vcs-f.png)

    - **Left**: name of the operator running on the NPU.
    - **Right**: name of the operator running on the GPU or CPU.
    - **Input**/**Output**: cosine similarity comparison result of the operator inputs/outputs. The value range is \[–1, +1\]. A value closer to  **1**  indicates higher similarity.

    As shown in the preceding figure, the operator inputs are basically the same, but their first outputs are remarkably different \(the cosine similarity is  **0.806927**, much less than  **0.98**\). This indicates that the operator may have an accuracy drop.

    > [!NOTE]NOTE
    > The list sorts operators with accuracy drop by execution sequence. As there are close ties between successive operators, analyze the top operator on the list.

7. Run the  [ni \(-n\) \[op_name\] -g \[graph\] -a \[attr\] -s \[save subgraph depth\]](precision_tool_command_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)  command to query the node information of a particular operator.

    ![](../figures/precision_ni-op.png)

    The  **ni**  command outputs the following information based on the passed operator name.

    1. Operator type. In this example, the operator type is Add.

        **PassName**  indicates that the operator is a fused operator, whose value indicates the fusion pattern name, and  **OriginOp**  indicates the base operators. The accuracy drop could be caused by operator fusion. In normal cases, any fusion bug should have been fixed in  [Fusion Exception Detection](fusion_exception_detection.md).

    2. Preliminary dump analysis result \(max/min/mean\).
    3. Subgraph of the specified depth with the current operator as the root, if the  **-s**  option is included. The following gives an example.

        ![](../figures/struct_diagram.png)

## Analysis Principles

Network-wide data comparison provides a layer-wise cumulative comparison report between network dump data and TensorFlow benchmark data. Even for networks without accuracy drop, errors caused by hardware differences are inevitable in the comparison result, and such errors will accumulate as the number of layers increases. Cosine similarity is a feasible metric to narrow down the operators with potential accuracy issues. A low cosine similarity always points to an accuracy bug while a high cosine similarity does not guarantee that the operator is 100% bug-free.

1. Determine whether an error operator is a custom operator based on the operator type.
    - For a custom operator, check that the implementation logic of the operator is consistent with that of the benchmark by inspecting the  [ni \(-n\) \[op_name\] -g \[graph\] -a \[attr\] -s \[save subgraph depth\]](precision_tool_command_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)  command output or the dump analysis report.
    - For a built-in CANN operator, if the operator input or output type is float16, you can switch the operator type to float32. You can try either of the following methods:
        1. \(Recommended\) Method 1: Modify the blocklist, trustlist, and graylist for the operator that uses the mixed precision mode. For details, see  [Modifying the Blocklist, Trustlist, and Graylist for Mixed Precision](../performance_tuning/mixed_precision_training.md#modifying-the-blocklist-trustlist-and-graylist-for-mixed-precision).
        2. Method 2: Use the  **keep_dtype_scope**  API to preserve the original precision for a selected operator.

            ```python
            from npu_bridge.npu_init import *
            with npu_scope.keep_dtype_scope():
              y = tf.multiply(x1,x2)
            ```

2. If the fault persists: please raise an issue in this source code repository.
