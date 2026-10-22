# Parameters That Will Be Deprecated in Later Versions

## enable_data_pre_proc

> In the current version, this parameter no longer takes effect. The system adaptively determines whether the GetNext operator is offloaded to the NPU.

Performance tuning. Specifies whether the GetNext operator is offloaded to the NPU. GetNext operator offload is a prerequisite for training iteration loop offload.

- True (default): offloaded. The prerequisite for GetNext operator offload is that the TensorFlow Dataset mode is used to read data.
- False: not offloaded.

Example:

```python
config = NPURunConfig(enable_data_pre_proc=True)
```

## variable_format_optimize

> This parameter is a function debugging switch and will be deprecated in later versions. Users are advised not to use it.

Performance tuning. Whether to enable variable format optimization.

- True: enabled.
- False: disabled.

To improve training efficiency, variables are converted to a format more suited for the AI processor during variable initialization performed by the network. In scenarios with special user requirements, this function can be disabled.

This parameter is left empty by default, indicating that the configuration is disabled.

Example:

```python
config = NPURunConfig(variable_format_optimize=True)
```

## op_debug_level

> This parameter will be deprecated in later versions. You are advised to use the [op_debug_config](./debugging.md#op_debug_config) parameter instead.

Operator debug enable.

- 0: disables operator debug.
- 1: Enables operator debug. TBE instruction mapping files are generated in the kernel_meta directory under the training script execution path, including operator CCE files (\*.cce), Python-CCE mapping files (\*_loc.json), .o files, and .json files. These files are used for AI Core error analysis with related tools.

  Note: For the Ascend 950PR&950DT products, no TBE instruction mapping files are generated.

- 2: Enables operator debug. TBE instruction mapping files are generated in the kernel_meta directory under the training script execution path, including operator CCE files (\*.cce), Python-CCE mapping files (\*_loc.json), .o files, and .json files. The compilation optimization of the CCE compiler is disabled and the CCE compiler debugging function is enabled (by setting the compiler option to -O0-g). These files are used for AI Core error analysis with related tools.

  Note: For the Ascend 950PR&950DT products, no TBE instruction mapping files are generated.

- 3: disables operator debug. The operator .o and .json files are retained in the kernel_meta folder in the training script execution directory.
- 4: disables operator debug. The operator binary (.o) and operator description file (.json) are **retained**, and a TBE instruction mapping file (.cce) and a UB fusion description file (\{$kernel_name\}_compute.json) are generated in the kernel_meta folder under the training script execution directory.

  For the Ascend 950PR&950DT products, neither TBE instruction mapping files nor UB fusion compute description files are generated.

**NOTICE:**

- If this option is set to 0 and op_debug_config is configured, the operator compilation directory kernel_meta is still generated in the current execution path during training. The content generated in the directory is subject to op_debug_config.
- You are advised to set this option to 0 or 3 for training. To locate AI Core errors, set this parameter to 1 or 2, which might compromise the network performance.
- If this option is set to 2 (the CCE compiler is enabled), it cannot be used together with the oom option in op_debug_config. Otherwise, an AI Core error is reported. The following is an example of the error message:

  ```text
  ...there is an aivec error exception, core id is 49, error code = 0x4 ...
  ```

- If this parameter is set to 2 (the CCE compiler is enabled), the size of the operator kernel file (\*.o file) increases. In dynamic shape scenarios, all possible scenarios are traversed during operator build, which may cause operator build failures due to large operator kernel files. In this case, 2 is not recommended.
  
  If the build failure is caused by the large operator kernel file, the following log is displayed:

  ```text
  message:link error ld.lld: error: InputSection too large for range extension thunk ./kernel_meta_xxxxx.o:(xxxx)
  ```

- If the value of this parameter is not 0, you can use the debug_dir parameter to specify the path for storing debugging-related process files.
- If this parameter is set to 0 and NPU_COLLECT_PATH is set, the operator compilation directory kernel_meta is still generated in the current path after the command is executed. If ASCEND_WORK_PATH is set, kernel_meta is generated in the path specified by the environment variable. For details about environment variables, see [Environment Variables](https://hiascend.com/en/document/redirect/CannCommunityEnvRef).
- When the debug function is enabled, if the model contains the following merged compute and communication (MC2) operators, the \*.o, \*.json, and \*.cce files of the operators are not generated in the operator build folder kernel_meta.

  ```text
  MatMulAllReduce
  MatMulAllReduceAddRmsNorm
  AllGatherMatMul
  MatMulReduceScatter
  AlltoAllAllGatherBatchMatMul
  BatchMatMulReduceScatterAlltoAll
  ```

This parameter is left empty by default, indicating that the configuration is disabled.

Example:

```python
config = NPURunConfig(op_debug_level=1)
```

## op_select_implmode

> This parameter will be deprecated in later versions. You are advised to use the [op_precision_mode](./performance_tuning.md#op_precision_mode) parameter instead.

Certain operators built in the NPU can be implemented in either high-precision or high-performance mode at model build time. Users can configure which implementation to use through this parameter.

- high_precision: high-precision implementation. In high-precision mode, Taylor's theorem or Newton's method is used to further improve operator precision with float16 input.
- high_performance: high-performance implementation. In high-performance mode, the optimal performance is implemented without affecting the network precision (float16).

This parameter is left empty by default, indicating that the configuration is disabled.

Example:

```python
config = NPURunConfig(op_select_implmode="high_precision")
```

## optypelist_for_implmode

> This parameter will be deprecated in later versions. You are advised to use the [op_precision_mode](./performance_tuning.md#op_precision_mode) parameter instead.

List of operator optypes (separated by commas) that use the mode specified by the op_select_implmode parameter. Currently, Pooling, SoftmaxV2, LRN, and ROIAlign operators are supported.

The `optypelist_for_implmode` parameter needs to be used in conjunction with the op_select_implmode parameter. Example:

```python
config = NPURunConfig(
    op_select_implmode="high_precision",
    optypelist_for_implmode="Pooling,SoftmaxV2")
```

This parameter is left empty by default, indicating that the configuration is disabled.

## dynamic_input

> This parameter will be deprecated in later versions and is not recommended for developers. The current version supports dynamic shape networks by default. For details, see the [jit_compile](./experiment_options.md#jit_compile) parameter.

Whether the current network input is a dynamic input. Possible values:

- True: dynamic input.
- False (default): static input.

Example:

```python
config = NPURunConfig(dynamic_input=True)
```

## dynamic_graph_execute_mode

> This parameter will be deprecated in later versions and is not recommended for developers. The current version supports dynamic shape networks by default. For details, see the [jit_compile](./experiment_options.md#jit_compile) parameter.

For dynamic input scenarios, this parameter sets the execution mode. That is, this option takes effect when dynamic_input is set to True. Possible values are:

dynamic_execute: dynamic graph compilation. In this mode, the shape range configured in dynamic_inputs_shape_range is used for compilation.

Example:

```python
config = NPURunConfig(dynamic_graph_execute_mode="dynamic_execute")
```

## dynamic_inputs_shape_range

> This parameter will be deprecated in later versions and is not recommended for developers. The current version supports dynamic shape networks by default. For details, see the [jit_compile](./experiment_options.md#jit_compile) parameter.

Shape range of each dynamic input. For example, if a graph has three inputs — two dataset inputs and one placeholder input — a configuration example is as follows:

```python
config = NPURunConfig(dynamic_inputs_shape_range="getnext:[128 ,3~5, 2~128, -1],[64 ,3~5, 2~128, -1];data:[128 ,3~5, 2~128, -1]")
```

Precautions:

- The dataset input is identified by "getnext" and the placeholder input is identified by "data". No other identifiers are allowed.
- The size of a static dimension is specified by a fixed number. The size range of a dynamic dimension is specified by using a tilde ("\~"). A dynamic dimension without size range specified is denoted by -1.
- For multiple inputs, for example, if there are three dataset inputs but only the second and third inputs have shape ranges while the first input is static, the static input shape must still be specified:

  ```python
  config = NPURunConfig(dynamic_inputs_shape_range="getnext:[3,3,4,10],[-1,3,2~1000,-1],[-1,-1,-1,-1]")
  ```

- For scalar inputs, you also need to fill in the shape range by using square brackets (\[\]). No space is allowed before "\[\]".
- If there are multiple getnext inputs or data inputs on the network, the input ordering must be preserved. For example:

  - If there are multiple dataset inputs on the network:

    ```python
    def func(x):
        x = x + 1
        y = x + 2
        return x,y
    dataset = tf.data.Dataset.range(min_size, max_size)
    dataset = dataset.map(func)
    ```

    Assume that the first input of the network is x (with shape range \[3\~5\]) and the second input is y (with shape range \[3\~6\]). When configuring the dynamic ranges in dynamic_inputs_shape_range, the ordering must be preserved.

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range ="getnext:[3~5],[3~6]")
    ```

  - If there are multiple placeholder inputs on the network:

    If the placeholder names are not specified, for example:

    ```python
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    ```

    The placeholder order is consistent with the definition order in the script. That is, the first input of the network is x (with shape range \[3\~5\]) and the second input is y (with shape range \[3\~6\]). When configuring the dynamic ranges in dynamic_inputs_shape_range, the ordering must be preserved.

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range= "data:[3~5],[3~6]")
    ```

    If the placeholder names are specified, for example:

    ```python
    x = tf.placeholder(tf.int32, name='b')
    y = tf.placeholder(tf.int32, name='a')
    ```

    The network inputs are sorted in alphabetical order of the name fields. That is, the first input of the network is y (with shape range \[3\~6\]) and the second input is x (with shape range \[3\~5\]). When configuring the dynamic ranges in dynamic_inputs_shape_range, the ordering must be preserved.

    ```python
    config = NPURunConfig(dynamic_inputs_shape_range = "data:[3~6],[3~5]")
    ```

    **NOTICE:**
    - For subgraphs with different input shapes, since dynamic_inputs_shape_range is a configuration property for a single graph, execution errors may occur. You are advised to use [set_graph_exec_config](../../npu_util/set_graph_exec_config.md) to support dynamic input scenarios.
    - If the placeholder names are not specified in the network script, the placeholders are named in the following format:

      xxx_0, xxx_1, xxx_2, ......
      
      The content following the underscore (_) is the sequence index of a placeholder in the network script. Placeholders are arranged in alphabetical order of the index. If the number of placeholders is greater than 10, the sequence is xxx_0 -> xxx_10 -> xxx_2 -> xxx_3. In the network script, the placeholder with index 10 is placed before the placeholder with index 2, causing the defined shape range to not match the input placeholder.

      To avoid this problem, when the number of input placeholders is greater than 10, you are advised to specify the placeholder names in the network script. In this case, the placeholders are named based on the specified names, to associate the shape ranges with the placeholder names.
    - This option cannot be used together with dynamic_dims. If both are configured, dynamic_dims takes precedence and this option is ignored.

## graph_memory_max_size

- In earlier versions, this parameter was used to specify the sizes of the network static memory and the maximum dynamic memory.
- In the current version, this parameter no longer takes effect. The system dynamically allocates memory based on the actual memory usage of the network.

## variable_memory_max_size

- In earlier versions, this parameter was used to specify the size of the variable memory.
- In the current version, this parameter no longer takes effect. The system dynamically allocates memory based on the actual memory usage of the network.
