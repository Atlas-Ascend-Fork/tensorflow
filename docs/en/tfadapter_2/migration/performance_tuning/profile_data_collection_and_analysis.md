# Profile Data Collection and Analysis

## Features

If basic tuning cannot achieve satisfactory performance, you can use the Profiling tool to collect profile data during training and analyze it to accurately locate software and hardware performance bottlenecks, thereby improving the performance analysis efficiency. The tool provides an economical solution for improving service performance.

By default, profile data is not collected during training. If you need to collect and parse profile data, see the procedure in this section.

The following describes the process of collecting and analyzing TensorFlow network profile data.

![](../figures/profiling_process.png)

1. Collect profile data.

    The profile data of the TensorFlow network can be collected globally or locally.

    - Global collection: collect the profile data of all behaviors executed by graphs. The data size is large.
    - Local collection: collect the profile data of specified subgraphs or steps.

    To collect data globally, you can either modify the training script and configure the  **enable_profiling**  parameter \(see  [Methods for Modifying the Training Script](#methods-for-modifying-the-training-script)\), or set the environment variable  **PROFILING_MODE**  \(see  [Using Environment Variables](#using-environment-variables)\).  **enable_profiling**  is prior to  **PROFILING_MODE**.

    To collect data locally, you can call the Profiler class in TF Adapter 1.x through the  **with**  statement and put the operations that require profile data collection into the Profiler class. For details, see  [Collecting Profile Data Locally](#collecting-profile-data-locally).

2. Parse and export profile data.

    Regardless of the collection mode, you can use the  **msprof**  command line to parse the profile data and export the parsing result to a specified directory. For details, see  [Parsing and Exporting Profile Data](#parsing-and-exporting-profile-data).

3. Analyze the profile data.

    You can analyze the timeline and summary files obtained by parsing the profile data to identify performance bottlenecks. For typical analysis examples, see  [Analyzing Profile Data](#analyzing-profile-data).

## Collecting Profile Data Globally

### Methods for Modifying the Training Script

Add  **profiling_config**  to the training script before initializing the NPU, to specify a tuning mode.

```python
import npu_device as npu
npu.global_options().profiling_config.enable_profiling=True
npu.global_options().profiling_config.profiling_options = '{"output":"/tmp/profiling","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
npu.open().as_default()
```

In the preceding command:

- **enable_profiling**: whether to enable profiling.
- **profiling_options**: profiling configuration options.
  - **output**: path for storing profile data. Create the specified directory in the training environment \(container or host\) in advance. The running user configured during installation must have the read and write permissions on this path. It can be either an absolute path or a relative path.
  - **task_trace**: task trace collection enable.
  - **training_trace**: iteration trace collection enable. If it is set to  **on**, both  **fp_point**  and  **bp_point**  need to be configured.
  - **aicpu**: whether to collect details about the AI CPU operator, such as the operator execution time and data copy time.
  - **fp_point**: start point of the forward propagated operator in iteration traces. This parameter is used to record the start timestamp of forward propagation. You can leave it empty to make the system obtain the values or manually obtain them.
  - **bp_point**: end point of the backward propagated operator in iteration traces. This parameter is used to record the end timestamp of backward propagation. You can leave it empty to make the system obtain the values or manually obtain them.
  - **aic_metrics**: AI Core, and AI Vector Core hardware information. The value  **PipeUtilization**  indicates the percentages of time taken by compute units and MTEs.

- For details about profiling configuration, see  [Profiling](../../apiref/npu-global_options/Profiling.md).

## Using Environment Variables

In addition to collecting profile data by modifying the training script, you can modify the corresponding environment variable in the startup script to enable profile data collection.

A configuration example is provided as follows:

```bash
# Enable profiling.
export PROFILING_MODE=true
# Configure profiling configuration options.
export PROFILING_OPTIONS='{"output":"/home/HwHiAiUser/output","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
```

For details about how to set the  **PROFILING_OPTIONS**  environment variable, see  _[Environment Variables](https://hiascend.com/document/redirect/CannCommunityEnvRef)_.

Note that the configuration item  **enable_profiling**  in the training script is prior to the environment variable  **PROFILING_MODE**.

## Collecting Profile Data Locally

Currently, TF Adapter 2.x does not support local collection of profile data. To collect profile data locally, you can use the  **compat.v1**  module to call the  **Profiler**  class in TF Adapter 1.x. That is, profile data sampling can be enabled only by running commands in the  **Profiler**  class.

For details about the Profiler class, see _[npu_bridge.profiler.profiler](../../../tfadapter_1/apiref/profiler/Profiler_constructor.md)_.

The following describes how to use the  **compat.v1**  module to call the Profiler class of TF Adapter 1.x to collect profile data locally.

1. Import the Profiler class.

    ```python
    import npu_device
    from npu_device.compat.v1.npu_init import *
    npu_device.compat.enable_v1() # Disable TF2 behavior and enable the TF1 compatibility mode.
    ```

2. Use the  **with**  statement to call the Profiler class and include the operations that require profile data collection in the Profiler class.

    In the following simple code snippet, a graph containing the Add operator is implemented and executed in a session. As  **sess.run \(add, ...\)**  is within the  **Profiler**  class, the L1 profile data and the ratios of compute metrics are collected. The profile data is stored in the current script execution path.

    ```python
    a = tf.placeholder(tf.int32, (None,None))
    b = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    add = tf.add(a, b)

    with tf.compat.v1.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
        result=sess.run(add, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    Currently, you can collect the profile data of specified steps by defining the specified step operations in the corresponding Profiler, as shown below.

    ```python
    a=tf.placeholder(tf.int32, (None,None))
    b=tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    d = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    add1 = tf.add(a, b)
    add2 = tf.add(c, d)
    add3 = tf.add(add1, add2)

    with tf.compat.v1.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="PipeUtilization", output_path = "/home/test/profiling_data"):
        for i in range(2):
          result=sess.run(add1, feed_dict={a: [[-20],[1]]})
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "/home/test/profiling_data"):
        for i in range(4):
          result=sess.run(add3, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    Pay attention to the following restrictions when using the Profiler class:

    - The  **Profiler**  class needs to be called using the  **with**  statement, and the profile data collection function takes effect in the corresponding scope.
    - The  **Profiler**  class can be called only in session mode.
    - The  **Profiler**  class cannot be nested.

        The following is an incorrect calling example:

        ```python
        with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
          with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
            sess.run(add)
        ```

    - The Profiler class cannot be used together with the configuration items  **enable_profiling**  and  **profiling_config**  in  ["global_options \> Profiling"](../../apiref/npu-global_options/Profiling.md), and the environment variables  **PROFILING_MODE**  and  **PROFILING_OPTIONS**.
    - The  **Profiler**  class does not support multi-thread calling.

# Parsing and Exporting Profile Data

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

# Analyzing Profile Data

Developers can identify performance bottlenecks by analyzing the timeline and summary files obtained by parsing the profile data with the Profiling tool.

The following describes only key profile data files. For details about more profile data files, see  _[Profiling Instructions](https://hiascend.com/en/document/redirect/CannCommunityToolProfiling)_.

- Timeline file:  **step_trace_\*.csv**

    The  **step_trace_\*.csv**  file records iteration trace data, including the duration of each iteration. The main fields are described as follows:

  - **Iteration Time**: computation time of an iteration, including the time of the FP/BP and Grad Refresh phases.
  - **FP to BP Time**: computation time of forward and backward propagation on the network.
  - **Iteration Refresh**: iteration trailing time.
  - **Data Aug Bound**: interval between two adjacent iterations.

    The following figure shows a data sample. To analyze data, select a proper iteration ID and model ID.

    **Figure  1**  Example of the  **step_trace_\*.csv**  file
    ![](../figures/step_trace_csv_file.png)

    According to the preceding example, data of the model whose ID is 1 is obviously different from the subsequent data. This model is the initialization graph. Data of the model whose ID is 11 is the real iterative computational graph. Therefore, you need to select data of model 11 for analysis. In addition, when the model ID is 11 and iteration ID is 1, you can find that the  **Data Aug Bound**  time is long because compilation is performed in this phase. Therefore, you need to select data after iteration 2 of model 11 for analysis.

- Summary files:  **op_statistic_\*.csv**  and  **op_summary_\*.csv**

    The  **op_statistic_\*.csv**  file records the AI Core and AI CPU operator execution times and time consumption. The  **op_summary_\*.csv**  file records the detailed AI Core and AI CPU operator data.

    Developers can preliminarily determine the time-consuming operators based on the  **op_statistic_\*.csv**  file, and then search for details of the time-consuming operators in the  **op_summary_\*.csv**  file to locate the minimum-granularity event.
