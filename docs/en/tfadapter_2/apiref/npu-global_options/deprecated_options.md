# Options That Will Be Deprecated in Later Versions

## op_select_implmode

> This parameter will be deprecated in later versions. You are advised to use the [op_precision_mode](./performance_tuning.md#op_precision_mode) parameter instead.

Certain operators built in the NPU can be implemented in either high-precision or high-performance mode at model build time. Users can configure which implementation to use through this parameter.

- high_precision: high-precision implementation. In high-precision mode, Taylor's theorem or Newton's method is used to further improve operator precision with float16 input.
- high_performance: high-performance implementation. In high-performance mode, the optimal performance is implemented without affecting the network precision (float16).

The default value is None, indicating that the configuration is disabled.

Example:

```python
npu.global_options().op_select_implmode="high_precision"
```

## optypelist_for_implmode

> This parameter will be deprecated in later versions. You are advised to use the [op_precision_mode](./performance_tuning.md#op_precision_mode) parameter instead.

List of operator optypes (separated by commas) that use the mode specified by the op_select_implmode parameter. Currently, Pooling, SoftmaxV2, LRN, and ROIAlign operators are supported.

This parameter needs to be used in conjunction with the op_select_implmode parameter. Example:

```python
npu.global_options().op_select_implmode="high_precision"
npu.global_options().optypelist_for_implmode="Pooling,SoftmaxV2"
```

The default value is None, indicating that the configuration is disabled.

## variable_format_optimize

> This parameter is a function debugging switch and will be deprecated in later versions. Users are advised not to use it.

Whether to enable variable format optimization.

- True: enabled.
- False: disabled.

To improve training efficiency, variables are converted to a format more suited for NPU execution during variable initialization performed by the network. In scenarios with special user requirements, this function can be disabled.

The default value is None, indicating that the configuration is disabled.

Example:

```python
npu.global_options().variable_format_optimize=True
```

## op_debug_level

> This parameter will be deprecated in later versions. You are advised to use [op_debug_config](./debugging.md#op_debug_config).

Operator debug enable.

- 0: disables operator debug.
- 1: Enables operator debug. TBE instruction mapping files are generated in the kernel_meta directory under the training script execution path, including operator CCE files (\*.cce), Python-CCE mapping files (\*_loc.json), .o files, and .json files. These files are used for AI Core error analysis with related tools.

  Note: For the Ascend 950PR&950DT products, no TBE instruction mapping files are generated.

- 2: Enables operator debug. TBE instruction mapping files are generated in the kernel_meta directory under the training script execution path, including operator CCE files (\*.cce), Python-CCE mapping files (\*_loc.json), .o files, and .json files. The compilation optimization of the CCE compiler is disabled and the CCE compiler debugging function is enabled (by setting the compiler option to -O0-g). These files are used for AI Core error analysis with related tools.

  Note: For the Ascend 950PR&950DT products, no TBE instruction mapping files are generated.

- 3: disables operator debug. The operator .o and .json files are retained in the kernel_meta folder in the training script execution directory.
- 4: disables operator debug. The operator binary (.o) and operator description file (.json) are retained, and a TBE instruction mapping file (.cce) and a UB fusion description file (`{$kernel_name}_compute.json`) are generated in the kernel_meta folder under the training script execution directory.

  Note:

  - For the Ascend 950PR&950DT products, neither TBE instruction mapping files nor UB fusion compute description files are generated.
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

The default value is None, indicating that the configuration is disabled.

Example:

```python
npu.global_options().op_debug_level=0
```

## graph_memory_max_size

In earlier versions, this parameter was used to specify the sizes of the network static memory and the maximum dynamic memory. In the current version, this parameter no longer takes effect. The system dynamically allocates memory based on the actual memory usage of the network.

## variable_memory_max_size

In earlier versions, this parameter was used to specify the size of the variable memory. In the current version, this parameter no longer takes effect. The system dynamically allocates memory based on the actual memory usage of the network.
