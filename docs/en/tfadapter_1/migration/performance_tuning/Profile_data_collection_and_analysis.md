# Profile Data Collection and Analysis

## Function Description

If basic tuning cannot achieve satisfactory performance, you can use the profiling tool to collect profile data during training and analyze it to accurately locate software and hardware performance bottlenecks, thereby improving the performance analysis efficiency. The tool provides an economical solution for improving service performance.

By default, profile data is not collected during training. If you need to collect and parse profile data, see the procedure in this section.

The following describes the process of collecting and analyzing TensorFlow network profile data.

![](../figures/profiling_overview.png)

1. Collect profile data.

    The profile data of the TensorFlow network can be collected globally or locally.

    - Global collection: collect the profile data of all behaviors executed by graphs. The data size is large.
    - Local collection: collect the profile data of a specified subgraph or step.

    To collect data globally, you can either modify the training script and configure **profiling_mode** (see  [Collecting Profile Data Globally (by Modifying the Training Script)](#collecting-profile-data-globally-by-modifying-the-training-script)), or set the environment variable **PROFILING_MODE** (see [Collecting Profile Data Globally (By Modifying Environment Variables)](#collecting-profile-data-globally-by-modifying-environment-variables)).  **profiling_mode**  is prior to  **PROFILING_MODE**.

    To collect data locally, you can call the Profiler class through the  **with**  statement and put the operations that require profile data collection into the Profiler class. For details, see  [Collecting Profile Data Locally (by Calling the Profiler Class)](#collecting-profile-data-locally-by-calling-the-profiler-class).

2. Parse and export profile data.

    Regardless of the collection mode, you can use the **msprof** command line to parse the profile data and export the parsing result to a specified directory. For details, see  [Parsing and Exporting Profile Data](#parsing-and-exporting-profile-data).

3. Analyze profile data.

    You can analyze the timeline and summary files obtained by parsing the profile data to identify performance bottlenecks. For typical analysis examples, see  [Analyzing Profile Data](#analyzing-profile-data).

## Collecting Profile Data Globally \(by Modifying the Training Script\)

### In Estimator Mode

#### Automated porting

1. Check whether **init_resource** exists in the ported script.
    - If it exists, modify it by referring to the following example. After the modification is complete, go to the next step.

        ```python
        if __name__ == '__main__':

          session_config = tf.ConfigProto(allow_soft_placement=True)
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # Enable profiling.
          custom_op.parameter_map["profiling_mode"].b = True
          # Collect only task trace data.
          custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
          # Collect task trace data and iteration trace data. You can collect only the task trace data. If the problem cannot be analyzed, collect the iteration trace data.
          # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

        Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md) can be configured in **config** of the **init_resource** function. For other functions, configure them in **run_config** of the **npu_run_config_init** function.

        > [!NOTE]NOTE
        >
        >- **profiling_mode**: whether to enable profiling.
        >- **output**: path for storing profile data. Create the specified directory in the training environment \(container or host\) in advance. The running user configured during installation must have the read and write permissions on this path. It can be either an absolute path or a relative path.
        >- **task_trace**: whether to enable task trace collection.
        >- **training_trace**: iteration trace collection enable. If it is set to  **on**, both  **fp_point**  and  **bp_point**  need to be configured.
        >- **aicpu**: whether to collect details about the AI CPU operator, such as the operator execution time and data copy time.
        >- **fp_point**: start point of the forward propagated operator in iteration traces. This parameter is used to record the start timestamp of forward propagation. You can leave it empty to make the system obtain the values or manually obtain them by referring to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md).
        >- **bp_point**: end point of the backward propagated operator in iteration traces. This parameter is used to record the end timestamp of backward propagation. You can leave it empty to make the system obtain the values or manually obtain them by referring to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md).
        >- **aic_metrics**: AI Core and AI Vector Core hardware information. The value  **PipeUtilization**  indicates the percentages of time taken by compute units and MTEs.
        >- For details about profiling configuration, see  [Profiling](../../apiref/session_config/Profiling.md).

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

3. Add the  **session_config**  configuration to enable profiling.

    ```python
    session_config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    # Enable profiling.
    custom_op.parameter_map["profiling_mode"].b = True
    # Collect only task trace data.
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
    # Collect task trace data and iteration trace data. You can collect only the task trace data first. If the problem cannot be analyzed, collect the iteration trace data.
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_secs=60*60*24)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
    ```

4. Run the training script again to collect profile data.

## Manual porting

You can try to collect task trace data by enabling  **task_trace**.

```python
from npu_bridge.npu_init import *

# enable_profiling: whether to enable profiling.
# output: path for storing profile data. Create the specified directory in the training environment (container or host) in advance. The running user configured during installation must have the read and write permissions on this path. It can be either an absolute path or a relative path.
# task_trace: task trace collection enable.
profiling_options = '{"output":"/home/test/output","task_trace":"on"}'
profiling_config = ProfilingConfig(enable_profiling=True, profiling_options= profiling_options)
session_config=tf.ConfigProto()

config = NPURunConfig(profiling_config=profiling_config, session_config=session_config)
```

\(Optional\) If the problem cannot be spotted, enable  **training_trace**  to collect iteration traces.

```python
from npu_bridge.npu_init import *

# enable_profiling: whether to enable profiling.
# output: path for storing profile data.
# task_trace: task trace collection enable
# training_trace: iteration trace collection enable
# fp_point: start point of the forward propagated operator in iteration traces, recording the start timestamp of forward propagation.
# bp_point: end point of the backward propagated operator in iteration traces, recording the end timestamp of backward propagation. fp_point and bp_point are used to compute the time used by forward and backward propagation.
profiling_options = '{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
profiling_config = ProfilingConfig(enable_profiling=True, profiling_options= profiling_options)
session_config=tf.ConfigProto(allow_soft_placement=True)

config = NPURunConfig(profiling_config=profiling_config, session_config=session_config)
```

Note that  **fp_point**  \(start point of the forward propagated operator in iteration traces\) and  **bp_point**  \(end point of the backward propagated operator in iteration traces\) are required for collecting iteration traces. You can leave them empty to make the system obtain the values or refer to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md)  to configure them when collection exceptions occur.

For details about related APIs, see  [ProfilingConfig Constructor](../../apiref/profiler/Profiler_constructor.md).

### In sess.run Mode

#### Automated porting

1. Check whether  **init_resource**  exists in the ported script.
    - If it exists, modify it by referring to the following example. After the modification is complete, go to the next step.

        ```python
        if __name__ == '__main__':

          session_config = tf.ConfigProto(allow_soft_placement=True)
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # Enable profiling.
          custom_op.parameter_map["profiling_mode"].b = True
          # Collect only task trace data.
          custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
          # Collect task trace data and iteration trace data. You can collect only the task trace data. If the problem cannot be analyzed, collect the iteration trace data.
          # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

        Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resoure**  function. For other functions, configure them in  **config_proto**  of the  **npu_config_proto**  function.

        > [!NOTE]NOTE
        >
        >- **profiling_mode**: whether to enable profiling.
        >- **output**: path for storing profile data. Create the specified directory in the training environment \(container or host\) in advance. The running user configured during installation must have the read and write permissions on this path. It can be either an absolute path or a relative path.
        >- **task_trace**: whether to enable task trace collection.
        >- **training_trace**: iteration trace collection enable. If it is set to  **on**, both  **fp_point**  and  **bp_point**  need to be configured.
        >- **aicpu**: whether to collect details about the AI CPU operator, such as the operator execution time and data copy time.
        >- **fp_point**: start point of the forward propagated operator in iteration traces. This parameter is used to record the start timestamp of forward propagation. You can leave it empty to make the system obtain the values or manually obtain them by referring to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md).
        >- **bp_point**: end point of the backward propagated operator in iteration traces. This parameter is used to record the end timestamp of backward propagation. You can leave it empty to make the system obtain the values or manually obtain them by referring to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md).
        >- **aic_metrics**: AI Core and AI Vector Core hardware information. The value  **PipeUtilization**  indicates the percentages of time taken by compute units and MTEs.
        >- For details about profiling configuration, see  [Profiling](../../apiref/session_config/Profiling.md).

    - If it does not exist, go to the next step.

2. Search for the  **npu_config_proto**  function in the ported script, find the run configuration parameter \(such as  **session_config**  in the following example\), and configure related parameters in the run configuration to enable  **task_trace**  data collection.

    ```python
    session_config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    # Enable profiling.
    custom_op.parameter_map["profiling_mode"].b = True
    # Collect only task trace data.
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
    # Collect task trace data and iteration trace data. You can collect only the task trace data first. If the problem cannot be analyzed, collect the iteration trace data.
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')
    config = npu_config_proto(config_proto=session_config)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        interaction_table.init.run()
    ```

#### Manual porting

You can try to collect task trace data by enabling  **task_trace**.

```python
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["profiling_mode"].b = True
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
  sess.run()
```

\(Optional\) If the problem cannot be spotted, enable  **training_trace**  to collect iteration traces.

```python
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["profiling_mode"].b = True
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
  sess.run()
```

Note that  **fp_point**  \(start point of the forward propagated operator in iteration traces\) and  **bp_point**  \(end point of the backward propagated operator in iteration traces\) are required for collecting iteration traces. You can leave them empty to make the system obtain the values or manually obtain them by referring to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md).

For details about related APIs, see  [Profiling](../../apiref/session_config/Profiling.md).

### In Keras Mode

1. Check whether  **init_resource**  exists in the ported script.
    - If it exists, modify it by referring to the following example. After the modification is complete, go to the next step.

        ```python
        if __name__ == '__main__':

          session_config = tf.ConfigProto(allow_soft_placement=True)
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # Enable profiling.
          custom_op.parameter_map["profiling_mode"].b = True
          # Collect only task trace data.
          custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
          # Collect task trace data and iteration trace data. You can collect only the task trace data. If the problem cannot be analyzed, collect the iteration trace data.
          # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

        Note that only the parameters supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resource**  function. For other functions, configure them in  **config**  of the  **set_keras_session_npu_config**  function.

        > [!NOTE]NOTE
        >
        >- **profiling_mode**: whether to enable profiling.
        >- **output**: path for storing profile data. Create the specified directory in the training environment \(container or host\) in advance. The running user configured during installation must have the read and write permissions on this path. It can be either an absolute path or a relative path.
        >- **task_trace**: whether to enable task trace collection.
        >- **training_trace**: iteration trace collection enable. If it is set to  **on**, both  **fp_point**  and  **bp_point**  need to be configured.
        >- **aicpu**: whether to collect details about the AI CPU operator, such as the operator execution time and data copy time.
        >- **fp_point**: start point of the forward propagated operator in iteration traces. This parameter is used to record the start timestamp of forward propagation. You can leave it empty to make the system obtain the values or manually obtain them by referring to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md).
        >- **bp_point**: end point of the backward propagated operator in iteration traces. This parameter is used to record the end timestamp of backward propagation. You can leave it empty to make the system obtain the values or manually obtain them by referring to  [How Do I Determine fp_point and bp_point?](../common_operation/fpbp_point_determination.md).
        >- **aic_metrics**: AI Core and AI Vector Core hardware information. The value  **PipeUtilization**  indicates the percentages of time taken by compute units and MTEs.
        >- For details about profiling configuration, see  [Profiling](../../apiref/session_config/Profiling.md).

    - If it does not exist, go to the next step.

2. Find  **set_keras_session_npu_config**  in the script and configure profiling parameters.

    ```python
    import tensorflow as tf
    import tensorflow.python.keras as keras
    from tensorflow.python.keras import backend as K
    from npu_bridge.npu_init import *

    config_proto = tf.ConfigProto(allow_soft_placement=True)
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    # Enable profiling.
    custom_op.parameter_map["profiling_mode"].b = True
    # Collect only task trace data.
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
    # Collect task trace data and iteration trace data. You can collect only the task trace data first. If the problem cannot be analyzed, collect the iteration trace data.
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')
    npu_keras_sess = set_keras_session_npu_config(config=config_proto)

    # Preprocess data...
    # Construct a model...
    # Build the model...
    # Train the model...
    ```

## Collecting Profile Data Globally (By Modifying Environment Variables)

In addition to collecting profile data by modifying the training script, you can also enable profiling by setting environment variables. A configuration example is provided as follows:

```bash
# Enable profiling.
export PROFILING_MODE=true
# Configure profiling configuration options.
export PROFILING_OPTIONS='{"output":"/home/test/output","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
```

For details about how to set the  **PROFILING_OPTIONS**  environment variable, see  Profile Data Collection  in  [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Note that the environment variable  **PROFILING_MODE**  takes lower priority than the  **profiling_mode**  configuration item in the training script.

## Collecting Profile Data Locally (by Calling the Profiler Class)

You can call the  [npu_bridge.profiler.profiler](../../apiref/profiler/Profiler_constructor.md)  class to locally collect profile data. That is, only the commands in the scope of the Profiler class can enable profile data collection.

The following describes how to call the Profiler class to enable profile data collection.

1. Import the Profiler class.

    ```python
    from npu_bridge.npu_init import *
    ```

2. Use the  **with**  statement to call the Profiler class and include the operations that require profile data collection in the Profiler class.

    In the following simple code snippet, a graph containing the Add operator is implemented and executed in a session. As  **sess.run \(add, ...\)**  is within the scope of the Profiler class, the L1 profile data is collected and the proportion of various computing metrics is calculated. The profile data is stored in the execution path of the current script.

    ```python
    a = tf.placeholder(tf.int32, (None,None))
    b = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    add = tf.add(a, b)

    with tf.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
        result=sess.run(add, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    Currently, you can collect the profile data of a specified step by defining the specified step operation in the corresponding Profiler, as shown below.

    ```python
    a=tf.placeholder(tf.int32, (None,None))
    b=tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    d = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    add1 = tf.add(a, b)
    add2 = tf.add(c, d)
    add3 = tf.add(add1, add2)

    with tf.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="PipeUtilization", output_path = "/home/test/profiling_data"):
        for i in range(2):
          result=sess.run(add1, feed_dict={a: [[-20],[1]]})
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "/home/test/profiling_data"):
        for i in range(4):
          result=sess.run(add3, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    For details about the restrictions on the Profiler class, see "Restrictions" in  [Profiler Constructor](../../apiref/profiler/Profiler_constructor.md).

## Parsing and Exporting Profile Data

The following uses the msprof command line as an example to describe how to parse and export profile data:

1. Switch to the path where the parsing tool is located.

    ```bash
    cd ${INSTALL_DIR}/tools/profiler/bin
    ```

    Replace  _$\{INSTALL_DIR\}_  with the CANN component directory. For example, if the installation is performed by the  **root**  user, the default file storage path is  **/usr/local/Ascend/cann**.

2. Run the following command to parse the profile data file:

    ```bash
    ./msprof --parse=on --output=/home/test/profiling_output
    ```

    **--output**  indicates the path for storing the profile data file set during profile data collection.

3. Export the profile data.

    ```bash
    ./msprof --export=on --output=/home/test/profiling_output
    ```

    For details about the Profiling tool, see  [Performance Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0001.html).

## Analyzing Profile Data

Developers can identify performance bottlenecks by analyzing the timeline and summary files obtained by parsing the profile data with the profiling tool.

The following describes key profile data files and analysis methods. For details about more profile data files, see  [Performance Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0001.html).

- Timeline file:  **step_trace_\*.csv**

    The  **step_trace_\*.csv**  file records iteration trace data, including the duration of each iteration. The main fields are described as follows:

  - **Iteration Time**: computation time of an iteration, including the time of the FP/BP and Grad Refresh phases.
  - **FP to BP Time**: computation time of forward and backward propagation on the network.
  - **Iteration Refresh**: iteration trailing time.
  - **Data Aug Bound**: interval between two adjacent iterations.

    The following figure shows a data sample. To analyze data, select a proper iteration ID and model ID.

    ![](../figures/step_trace_csv_file.png)

    According to the preceding example, data of the model whose ID is 1 is obviously different from the subsequent data. This model is the initialization graph. Data of the model whose ID is 11 is the real iterative computational graph. Therefore, you need to select data of model 11 for analysis. In addition, when the model ID is 11 and iteration ID is 1, you can find that the  **Data Aug Bound**  time is long because compilation is performed in this phase. Therefore, you need to select data after iteration 2 of model 11 for analysis.

- Summary files:  **op_statistic_\*.csv**  and  **op_summary_\*.csv**

    The  **op_statistic_\*.csv**  file records the AI Core and AI CPU operator execution times and time consumption. The  **op_summary_\*.csv**  file records the detailed AI Core and AI CPU operator data.

    Developers can preliminarily determine the time-consuming operators based on the  **op_statistic_\*.csv**  file, and then search for details of the time-consuming operators in the  **op_summary_\*.csv**  file to locate the minimum-granularity event.
