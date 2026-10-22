# Network Accuracy Comparison

## Overview

If the accuracy problem does not happen in the steps above, dump the compute result of each operator during the training process and compare the dump data with that of each benchmark operator \(such as the TensorFlow equivalents\) to quickly spot the faulty operators. The major steps are described as follows.

![](../figures/network_data_compare.png)

## Prerequisites

1. Floating-point exceptions have been excluded, and the overflow/underflow detection function has been disabled.
2. Fusion issues exceptions have been excluded, and the fusion switch has been restored to on.
3. You have completed  [One-Click Accuracy Analyzer Deployment](accuracy_analyzer_deployment.md).

4. All random operations for image preprocessing have been disabled in your training script. Failure to do so will result in unavailable comparison result due to inconsistent input data.

## Dumping Benchmark Data on GPU/CPU

- Before obtaining the dump data or .npy data of an original TensorFlow 2.x training network, a complete, executable, standard TensorFlow model training project is required.
- Install the debugger  **tfdbg_ascend**  of TensorFlow 2._x_. For details, see  [tfdbg_ascend README](https://gitee.com/ascend/tools/tree/master/tfdbg_ascend).
- Disable all random functions in the script, including but not limited to shuffle operations on datasets, random initialization of parameters, and implicit random initialization of some operators \(such as the dense operator\). Ensure that all parameters in your script are not initialized randomly.

You can use the TensorFlow debugger \(**tfdbg_ascend**\) to generate .npy files. The major steps are as follows:

1. Modify the configuration in the TensorFlow training script for model calling. The sample code is as follows:

    Sample 1:

    1. Import the debug plugin.

        ```python
        import tfdbg_ascend as dbg
        ```

    2. Add the following code before the training startup code of each step. For example, to dump the data of the fifth step, add the code as follows:

        ```python
              dbg.disable()
              if current_step == 5:
                  dbg.enable()
                  dbg.set_dump_path("home/test/gpu_dump")
        ```

    Sample 2:

    1. Import the debug plugin.

        ```python
        import tfdbg_ascend as dbg
        ```

    2. Dump the data of the fourth step \(example\). If you do not configure  **dbg.enable**, the dump function is enabled by default. If you do not specify the dump path, dump files are saved in the path where the training script is located by default.

        ```python
        class DumpConfig(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
            def on_batch_begin(self, batch, logs={}):
                if batch == 4:
                    dbg.enable()
                    dbg.set_dump_path("/home/test/gpu_dump")
                else:
                    dbg.disable()
        ```

    3. Register the callback functions.

        ```python
        # define callbacks
        callbacks = [
            ModelCheckpoint(
                f'models/model_epochs-{epochs}_batch-{batch_size}_loss-{loss_function}_{Mask2FaceModel.get_datetime_string()}.h5'),
            LossHistory(batch_size),
            DumpConfig()
        ]

        # fit the model
        history = self.model.fit(train_dataset, validation_data=valid_dataset, epochs=1, callbacks=callbacks, verbose=2)
        ```

2. Execute the training script. After the training job is stopped, the .npy files are generated in the specified directory.
3. Check that names of the generated .npy files comply with the naming rules, as shown in the figure below.

   ![](../figures/query_npy_file.png)

    > [!NOTE]NOTE
    >- An .npy file is named in the format  **_\{op_name\}_._\{output_index\}_._\{timestamp\}_.npy**, where  _**op_name**_  must comply with the  **A-Za-z0-9_-**  regular expression,  _**timestamp**_  must comply with the  **\[0-9\]\{1,255\}**  regular expression, and  _**output_index**_  must be a digit ranging from 0 to 9.
    >- If the name of an .npy file exceeds 255 characters due to a long operator name, comparison of this operator is not supported.

## Dumping User Model on the NPU

Perform the following operations in the NPU training environment. Pay attention to the following points before dumping data:

Generally, dump of the first step is enough for comparison and analysis. To avoid inaccurate comparison caused by random weights, enable checkpoints saving before training. If you find an accuracy issue with a particular step, resume the training process from the checkpoint closest to the particular step.

1. Modify the  **config.py**  file in the  **precision_tool/lib/config**  directory and specify the step of the data to be dumped.

    ```python
    # Dump data with a specific step. To dump the input layer, retain the default value. Modify the value if you want to specify steps, for example, '0|5|10'.
    TF_DUMP_STEP = '0'
    ```

    If  **TF_DUMP_STEP**  is not set, dump data of all iterations is collected.

2. Edit the original training script to enable dumping.

    With the following script, both dump data and dump graphs are generated.

    ```python
    import precision_tool.tf_config as npu_tf_config
    npu_tf_config.npu_device_dump_config(npu_device, action='dump')
    ```

    > [!NOTE]NOTE
    > In addition to this method, you can find another mode to collect dump data in  [Accuracy Analyzer](https://hiascend.com/en/document/redirect/CannCommunityToolAccucacy). However, the configuration is complex, and you need to manually extract the dump data and save it to the required directory for analysis. Note that the two modes are mutually exclusive.

3. Run training. The dump graph and dump data files of GE are generated in the  **precision_data/npu/debug_0**  directory.

    For details about subsequent data analysis, see  [Comparing Dump Data](#comparing-dump-data).

## Comparing Dump Data

Accuracy analysis depends on the  **atc**  and  **msaccucmp.py**  tools in the CANN package. Perform the following operations in the CANN development environment:

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

2. Install the Python dependencies.

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
    # This path depends on the atc and msaccucmp.py tools in the CANN package. Set it to the CANN package installation directory.
    # By default, the CANN package is installed in /usr/local/Ascend. You can retain the path or replace the path as needed.
    CMD_ROOT_PATH = '/usr/local/Ascend'
    ```

4. Start the precision_tool command line.

    **python3 ./precision_tool/cli.py**

    Enter the command line interface:

    **PrecisionTool \>**

5. Run the  [ac -l \[limit_num\] \(-c\)](precision_tool_ommand_ref.md#ac--l-limit_num--c)  command for network comparison.

    **PrecisionTool \> ac -c**

    The time consumption varies depending on the data size.

    The comparison result is saved in CSV format in the  **precision_data/temp/vector_compare**  directory.

    ![](../figures/precision_ac-c.png)

    You can directly inspect the CSV file. For details, see  [Network Accuracy Comparison Result File](network_accuracy_comparison_result_file.md).

6. \(Optional\) Run the  [vcs -f \[file_name\] -c \[cos_sim_threshold\] -l \[limit\]](precision_tool_ommand_ref.md#vcs--f-file_name--c-cos_sim_threshold--l-limit)  command to narrow down the operators with potential accuracy issues.

    By default, the  **vcs**  command returns operators with cosine similarity values less than  **0.98**. The threshold can be user-defined by using the  **-c**  argument.

    ![](../figures/precision_vcs-f.png)

    - **Left**: name of the operator running on the NPU.
    - **Right**: name of the operator running on the GPU or CPU.
    - **Input**/**Output**: cosine similarity comparison result of the operator inputs/outputs. The value range is \[–1, +1\]. A value closer to  **1**  indicates higher similarity.

    As shown in the preceding figure, the operator inputs are basically the same, but their first outputs are remarkably different \(the cosine similarity is  **0.806927**, much less than  **0.98**\). This indicates that the operator may have an accuracy drop.

    > [!NOTE]NOTE
    > The list sorts operators with accuracy drop by execution sequence. As there are close ties between successive operators, analyze the top operator on the list.

7. Run the  [ni \(-n\) \[op_name\] -s \[save sub graph deep\]](precision_tool_ommand_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)  command to query the node information of a particular operator.

    ![](../figures/precision_ni-n.png)

    The  **ni**  command outputs the following information based on the passed operator name.

    1. Operator type. In this example, the operator type is Add.

        **PassName**  indicates that the operator is a fused operator, whose value indicates the fusion pattern name, and  **OriginOp**  indicates the base operators. The accuracy drop could be caused by operator fusion. In normal cases, any fusion bug should have been fixed in  [Floating-Point Exception Detection](floating-point_exception_detection.md).

    2. Preliminary dump analysis result \(max/min/mean\).
    3. Subgraph of the specified depth with the current operator as the root, if the  **-s**  option is included. The following gives an example.

       ![](../figures/structural_drawing.png)

## Analysis Principles

Network comparison provides a layer-by-layer cumulative comparison report between the dumped network data and the TensorFlow benchmark data. Due to hardware differences, networks have inherent numerical deviations that accumulate as the number of layers increases. Even for networks with normal accuracy, minor numerical discrepancies exist. Cosine similarity is generally used for preliminary screening of suspicious operators. Note that high cosine similarity does not necessarily indicate the absence of issues, while low cosine similarity usually suggests potential issues. The accuracy comparison results can provide a general direction for analysis.

1. Determine whether an error operator is a custom operator based on the operator type.
    - For a custom operator, check that the implementation logic of the operator is consistent with that of the benchmark by inspecting the  [ni \(-n\) \[op_name\] -s \[save sub graph deep\]](precision_tool_ommand_ref.md#ni--n-op_name--g-graph--a-attr--s-save-subgraph-depth)  command output or the dump analysis report.
    - For a built-in CANN operator, if the operator input or output type is float16, you can switch the operator type to float32. You can try either of the following methods:
        1. \(Recommended\) Method 1: Use  [modify_mixlist](../../apiref/npu-global_options/accuracy_tuning.md#modify_mixlist) to modify the blocklist, trustlist, and graylist for the operator that uses the mixed precision mode.
        2. Method 2: Use the  [npu.keep_dtype_scope](../../apiref/npu-keep_dtype_scope.md)  API to preserve the original precision for a selected operator.

            ```python
            import npu_device as npu
            with npu.keep_dtype_scope():
                v = tf.add(1, 1)
            ```

2. If the fault persists, please raise an issue in this source code repository.
