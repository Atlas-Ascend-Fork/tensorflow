# Dump Data Collection

## Overview

If the training network's accuracy does not meet your expectation, you can collect the compute result \(dump data\) of each operator and then compare it with that of the third-party operator \(such as a TensorFlow equivalent\) to facilitate location of operator accuracy issues. Currently, the following dump modes are supported:

- **input**: dumps operator inputs.
- **output**: dumps operator outputs.
- **all**: dumps both operator inputs and outputs.

> [!CAUTION]NOTICE
>By default, the dump data of operators is not collected during the training process. If you need to collect and analyze the data, you can follow the method described in this section or refer to  [Network-wide Accuracy Comparison](../accuracy_debugging/network_accuracy_comparison.md)  for a user-friendly one-click solution that simplifies the collection and analysis process, which is recommended to use.

## Precautions

- Currently, all iterations can be dumped. You can specify the iterations to be dumped. If the training dataset is large, the dump data volume of each iteration can reach about dozens of GB. You are advised to control the number of iterations.
- Data dump \(**enable_dump**\) is mutually exclusive with overflow/underflow detection \(**enable_dump_debug**\).
- The dump data of the AI Core, AI CPU, and collective communication operators can be collected.
- You are advised to retain only the  **sess.run**  code of the computing process and delete unnecessary  **sess.run**  code, for example,  **sess.run\(global_step\)**. Otherwise, an exception may occur during data dump.

## Collection in Estimator Mode

- Automated porting
    1. Check whether  **init_resource**  exists in the ported script.
        - If it exists, refer to the following example to pass the  **config**  configuration to the  **init_resource**  function.

            ```python
            if __name__ == '__main__':

              session_config = tf.ConfigProto(allow_soft_placement=True)
              custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
              custom_op.name = "NpuOptimizer"
              # enable_dump: whether to enable the data dump function.
              custom_op.parameter_map["enable_dump"].b = True
              # dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
              custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
              # dump_step: iterations to dump.
              custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
              # dump_mode: dump mode, selected from input, output, and all.
              custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
              # dump_layer: operator to be dumped. The value is the operator name. Separate multiple operator names using spaces. If this parameter is not set, all operators are dumped by default.
              custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes("nodename1 nodename2 nodename3")

              (npu_sess, npu_shutdown) = init_resource(config=session_config)
              tf.app.run()
              shutdown_resource(npu_sess, npu_shutdown)
              close_session(npu_sess)
            ```

            Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resource**  function. For other functions, configure them in  **run_config**  of the  **npu_run_config_init**  function.

        - If it does not exist, go to the next step.

    2. Search for  **npu_run_config_init**  in the ported script and find the run configuration function, such as  **run_config**  in the example.

        If the  **session_config**  parameter does not exist in the run configuration function, add the parameter according to the following example. If the  **session_config**  parameter exists, go to the next step.

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)

        run_config = tf.estimator.RunConfig(
            train_distribute=distribution_strategy,
            session_config=session_config,
            save_checkpoints_secs=60*60*24)

        classifier = tf.estimator.Estimator(
            model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
        ```

    3. Modify the  **session_config**  configuration and add related dump parameters.

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        # enable_dump: whether to enable the data dump function.
        custom_op.parameter_map["enable_dump"].b = True
        # dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
        # dump_step: iterations to dump.
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
        # dump_mode: dump mode, selected from input, output, and all.
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        # dump_layer: operator to be dumped. The value is the operator name. Separate multiple operator names using spaces. If this parameter is not set, all operators are dumped by default.
        custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes("nodename1 nodename2 nodename3")

        run_config = tf.estimator.RunConfig(
            train_distribute=distribution_strategy,
            session_config=session_config,
            save_checkpoints_secs=60*60*24)

        classifier = tf.estimator.Estimator(
            model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
        ```

- Manual porting

    In  **Estimator**  mode, use  **dump_config**  in  **NPURunConfig**  to collect dump data. Before  **NPURunConfig**  is created, instantiate a  **DumpConfig**  class for dump configuration, including the dump path, iterations to dump, and dump mode.

    For details about each field in the constructor of the  **DumpConfig**  class, see  [DumpConfig Constructor](../../apiref/npu_config/dumpconfig_constructor.md).

    ```python
    from npu_bridge.npu_init import *

    # dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
    # enable_dump: whether to enable the data dump function.
    # dump_step: iterations to dump.
    # dump_mode: dump mode, selected from input, output, and all.
    dump_config = DumpConfig(enable_dump=True, dump_path = "/home/test/output", dump_step="0|5|10", dump_mode="all")

    session_config=tf.ConfigProto(allow_soft_placement=True)

    config = NPURunConfig(
      dump_config=dump_config,
      session_config=session_config
      )
    ```

## Collection in sess.run Mode

- Automated porting
    1. Check whether  **init_resource**  exists in the ported script.
        - If it exists, refer to the following example to pass the  **config**  configuration to the  **init_resource**  function.

            ```python
            if __name__ == '__main__':

              session_config = tf.ConfigProto(allow_soft_placement=True)
              custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
              custom_op.name = "NpuOptimizer"
              # enable_dump: whether to enable the data dump function.
              custom_op.parameter_map["enable_dump"].b = True
              # dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
              custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
              # dump_step: iterations to dump.
              custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
              # dump_mode: dump mode, selected from input, output, and all.
              custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
              # dump_layer: operator to be dumped. The value is the operator name. Separate multiple operator names using spaces. If this parameter is not set, all operators are dumped by default.
              custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes("nodename1 nodename2 nodename3")

              (npu_sess, npu_shutdown) = init_resource(config=session_config)
              tf.app.run()
              shutdown_resource(npu_sess, npu_shutdown)
              close_session(npu_sess)
            ```

            Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resource**  function. For other functions, configure them in  **config_proto**  of the  **npu_config_proto**  function.

        - If it does not exist, go to the next step.

    2. Search for  **npu_config_proto**  in the ported script, find the run configuration parameter \(such as  **session_config**  in the following example\), and add the dump configuration to the run configuration, as shown in the following:

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes("nodename1 nodename2 nodename3")
        config = npu_config_proto(config_proto=session_config)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            interaction_table.init.run()
        ```

- Manual porting

    In  **sess.run**  mode, you can collect dump data using session configuration options such as  **enable_dump**,  **dump_path**,  **dump_step**, and  **dump_mode**. For details about the parameters, see  [Accuracy comparison](../../apiref/session_config/accuracy_comparison.md).

    ```python
    config = tf.ConfigProto(allow_soft_placement=True)

    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True

    # enable_dump: whether to enable the data dump function.
    custom_op.parameter_map["enable_dump"].b = True
    # dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
    # dump_step: iterations to dump.
    custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
    # dump_mode: dump mode, selected from input, output, and all.
    custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    # dump_layer: operator to be dumped. The value is the operator name. Separate multiple operator names using spaces. If this parameter is not set, all operators are dumped by default.
    custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes("nodename1 nodename2 nodename3")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    with tf.Session(config=config) as sess:
      print(sess.run(cost))
    ```

## In tf.keras Mode

- Automated porting
    1. Check whether  **init_resource**  exists in the ported script.
        - If it exists, refer to the following example to pass the  **config**  configuration to the  **init_resource**  function.

            ```python
            if __name__ == '__main__':

              session_config = tf.ConfigProto(allow_soft_placement=True)
              custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
              custom_op.name = "NpuOptimizer"
              # enable_dump: whether to enable the data dump function.
              custom_op.parameter_map["enable_dump"].b = True
              # dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
              custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
              # dump_step: iterations to dump.
              custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
              # dump_mode: dump mode, selected from input, output, and all.
              custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
              # dump_layer: operator to be dumped. The value is the operator name. Separate multiple operator names using spaces. If this parameter is not set, all operators are dumped by default.
              custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes("nodename1 nodename2 nodename3")

              (npu_sess, npu_shutdown) = init_resource(config=session_config)
              tf.app.run()
              shutdown_resource(npu_sess, npu_shutdown)
              close_session(npu_sess)
            ```

            Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resource**  function. For other functions, configure them in  **config**  of the  **set_keras_session_npu_config**  function.

        - If it does not exist, go to the next step.

    2. Search for  **set_keras_session_npu_config**  in the script, find the run configuration, for example,  **config_proto**, and add the dump configuration to the run configuration, as shown in the following:

        ```python
        import tensorflow as tf
        import tensorflow.python.keras as keras
        from tensorflow.python.keras import backend as K
        from npu_bridge.npu_init import *

        config_proto = tf.ConfigProto(allow_soft_placement=True)
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        # enable_dump: whether to enable the data dump function.
        custom_op.parameter_map["enable_dump"].b = True
        # dump_path: dump path.
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
        # dump_step: iterations to dump.
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
        # dump_mode: dump mode, selected from input, output, and all.
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        # dump_layer: operator to be dumped. The value is the operator name. Separate multiple operator names using spaces. If this parameter is not set, all operators are dumped by default.
        custom_op.parameter_map["dump_layer"].s = tf.compat.as_bytes("nodename1 nodename2 nodename3")
        npu_keras_sess = set_keras_session_npu_config(config=config_proto)

        # Preprocess data...
        # Construct a model...
        # Build the model...
        # Train the model...
        ```

- Manual porting

    The configuration method is similar to that of manual porting in  **sess.run**  mode. For details, see  [Collection in sess.run Mode](#collection-in-sessrun-mode).

## Collection on a Rank in Distributed Training

For distributed data parallel, if you need to reduce the amount of dump data, you can specify only one rank to collect dump data. The following uses an Estimator training script for demonstration.

```python
if int(os.getenv('RANK_ID')) == 7:
    dump_flag = True
else:
    dump_flag = False
dump_config = DumpConfig(enable_dump=dump_flag, dump_path="/home/data_dump", dump_step="20", dump_mode="output")
```

## Performing Training and Dump Data Generation

After dump data collection is enabled, a dump file of the computational graph \(basic dump without data such as weights; only the graph optimized and compiled by the GE is dumped\) is automatically generated in the current execution directory during script execution. This computational graph file is used to search for dump data files in the follow-up accuracy analysis. You can also use the environment variable  **DUMP_GRAPH_PATH**  to specify the path for storing the dump graph file. The following is an example:

```bash
export DUMP_GRAPH_PATH=/home/dumpgraph
```

1. Start training to generate dump graph files and dump data files.

    - The dump graph file is generated in the  **\$\{DUMP_GRAPH_PATH\}/pid_$\{pid\}_deviceid_$\{deviceid\}**  directory and prefixed with  **ge**.
    - The dump data file is generated in the directory specified by  **dump_path**, that is,  _**\{dump_path\}/\{time\}/\{deviceid\}/\{model_name\}/\{model_id\}/\{data_index\}**_. For example, if  **dump_path**  is set to  **/home/test/output**, the dump data file is stored in the  **/home/test/output/20200808163566/0/ge_default_20200808163719_121/11/0**  directory.

        NOTE:
        - If the training dataset is large, the dump data volume of each iteration can reach about dozens of GB or even more. You are advised to control the number of iterations to one.
        - In the multi-device training scenario where more than one Ascend AI Processor is used, since the processes are not started at the same time as defined in the training script, multiple timestamp directories are generated when data is dumped.
        - When the command is executed in a Docker, the generated data is stored in the Docker.
        - No dump data will be generated for the following operators during graph execution:
          - Operators confirmed not to execute on the device before graph execution, including conditional operators \(such as if, while, for, and case\), data operators \(such as Data, RefData, and Const\), and data flow operators \(such as StackPush, StackPop, Concat, and Split\).
          - Operators marked by the GE during graph optimization to skip execution on the device. For such operators, the  **_no_task**  attribute of  **attr**  in the dump graph is  **true**.
            - Operators located on unreachable execution branches in the graph.

        | Path Key | Description | Remarks |
        | --- | --- | --- |
        | dump_path | Dump path in the training script. (If it is set to a relative path, the dump path is a full path after combination.) | -- |
        | time | Dump time. | Format: YYYYMMDDHHMMSS |
        | deviceid | Device ID. | -- |
        | model_name | Subgraph name. | If the model_name directory contains more than one folder, dump data in the folder with the same name as the computational graph is used.<br>Periods (.), forward slashes (/), backslashes (\), and spaces in model_name are replaced with underscores (_). |
        | model_id | Subgraph ID. | -- |
        | data_index | Iterations to dump. | If dump_step is specified, data_index and dump_step are the same. If not, data_index starts at 0 and is incremented by 1 with each dump. |
        | dump_file | Format: {op_type}.{op_name}.{taskid}.{stream_id}.{timestamp}. If the length of a file name formatted as required exceeds the OS file name length limit (generally 255 characters), the dump file is renamed a string of random digits. For details about the mapping, see the mapping.csv file in the same directory. | Periods (.), forward slashes (/), backslashes (\), and spaces in op_type or op_name are replaced with underscores (_). |

    There are a large number of dump graph files prefixed with  **ge**, and multiple folders may exist in  **model_name**  of the dump data folder. But you only need to find the computational graph file and the folder whose  **model_name**  is the name of the computational graph. The following methods are provided to quickly find the corresponding files.

2. Select a computational graph file.

    The following two methods are provided:

    - Method 1: Search for the keyword  **Iterator**  in all dump files suffixed with  **_Build.txt**. Record the name of the computational graph file, which will be used in accuracy analysis.

        ```bash
        grep Iterator *_Build.txt
        ```

         ![](../figures/dump_build.png)

        As shown in the preceding figure, the  **ge_proto_00292_Build.txt**  file is the desired computational graph file.

    - Method 2: Save the TensorFlow model as a .pb file. Check the file, select the name of a compute operator as the keyword, and search for the keyword in all dump files. The needed computational graph file is the one with the keyword.

3. Obtain a dump data file.
    1. Open the computational graph file found in step2 and record the value of the  **name**  field in the first graph. In the following example, record the value  **"ge_default_20240613143502_1"**.

        ```text
        graph {
          name: "ge_default_20240613143502_1"
          op {
            name: "atomic_addr_clean0_71"
            type: "AtomicAddrClean"
            attr {
              key: "_fe_imply_type"
              value {
                i: 6
              }
            }
          }
        }
        ```

    2. Go to the directory for storing the dump file named after the timestamp. The following folders exist in the directory:

        ![](../figures/dump_ge_default.png)

    3. Find the folder named after the computational graph name, for example,  **ge_default_20240613143502_1**. The files in the folder are the required dump data files.

        ![](../figures/dump_ge_fault_data.png)

## Model Accuracy Analysis

You can use Model Accuracy Analyzer to analyze the dump data by referring to  Extended Functions \> Viewing Dump Files  in  [Accuracy Analyzer](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/ModelAccuracyAnalyzer/atlasaccuracy_16_0001.html).
