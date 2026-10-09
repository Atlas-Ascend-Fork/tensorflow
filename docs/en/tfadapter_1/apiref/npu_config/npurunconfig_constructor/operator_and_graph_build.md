# Operator and Graph Compilation

## op_compiler_cache_mode

Disk cache mode for operator building. enable is the default value.

- enable (default): disk cache mode enabled. The operator compilation information is cached to the disk, which can be reused by operators with the same compilation parameters, improving compilation efficiency.
- force: cache mode enabled. This mode deletes the existing cache, then recompiles the operators and adds them to the cache. For example, for Python changes, dependency library changes, or repository changes after operator optimization, you need to set this parameter to force to clean up the existing cache and then change it to enable to prevent the cache from being forcibly refreshed during each compilation. Note that you are not advised to set the force option for parallel program compilation. Otherwise, the cache used by other models may be cleaned up, causing compilation failures.
- disable: disk cache mode disabled.

Notes:

- When enabling the operator compilation cache function, you can configure the path for storing the operator compilation cache file by using op_compiler_cache_dir.
- disable and force are recommended for publishing the final model.
- If op_debug_level is set to a non-zero value, the op_compiler_cache_mode configuration is ignored, the operator compilation cache function is disabled, and all operators are recompiled.
- If op_debug_config is not empty and the op_debug_list field is not configured, the op_compiler_cache_mode configuration is ignored, the operator compilation cache function is disabled, and all operators are recompiled.
- If op_debug_config is not empty, the op_debug_list field is configured, and op_compiler_cache_mode is set to enable or force, the operators in the list are recompiled, and the operator compilation cache function is enabled for operators that are not in the list. However, operators that are not in the list will not be recompiled.
- When the operator compilation cache function is enabled, the default disk space allocated for cache files is 500 MB. If disk space becomes insufficient, cache files are deleted and 50% of the cache space is reserved by default. You can also customize the disk space allocated for cache files and the percentage of cache space to retain as follows:

1. Using the op_cache.ini configuration file.

   After the operator is compiled, the op_cache.ini file is automatically generated in the directory specified by op_compiler_cache_dir. You can use this file to set the disk space allocated for cache files and the percentage of cache space to retain. If the op_cache.ini file does not exist, manually create it.

   Add the following information to the op_cache.ini file:

   ```text
   # Configure the file format (required). The automatically generated file contains the following information by default. When manually creating a file, enter the following information:
   [op_compiler_cache]
   # Limit the disk space of the cache file on the AI processor (unit: MB).
   max_op_cache_size=500
   # When the disk space is insufficient, set the percentage of cache files to retain. Value range: [1, 100] (%). For example, setting it to 80 means that when disk space becomes insufficient, 80% of the cache files will be retained and the rest will be deleted.
   remain_cache_size_ratio=80
   ```

   - The op_cache.ini file takes effect only when the values of max_op_cache_size and remain_cache_size_ratio in the preceding file are valid.
   - When the size of the compilation cache file exceeds the configured value of max_op_cache_size and the cache file has not been accessed for more than half an hour, the cache file will be aged out. (Operator compilation will not be interrupted if the cache file size exceeds the limit. Therefore, if max_op_cache_size is set too small, the actual compilation cache file size may exceed the configured value.)
   - To disable the compilation cache aging function, set max_op_cache_size to -1. In this case, the access time is not updated when the operator cache is accessed, the operator compilation cache is not aged, and the default disk space of 500 MB is used.
   - If multiple users use the same cache path, the configuration file affects all users.

2. Using environment variable ASCEND_MAX_OP_CACHE_SIZE.

   You can use the environment variable ASCEND_MAX_OP_CACHE_SIZE to limit the disk space for cache files under an AI processor. When the compilation cache space reaches the value set by ASCEND_MAX_OP_CACHE_SIZE and a cache file has not been accessed for more than half an hour, the cache file will be aged out. ASCEND_REMAIN_CACHE_SIZE_RATIO can be used to set the percentage of cache space to retain. For details about environment variables, see in [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

   To disable the compilation cache aging function, set ASCEND_MAX_OP_CACHE_SIZE to -1.

  If both the op_cache.ini file and environment variables are configured, the configuration items in op_cache.ini take precedence. If neither is configured, the system uses the default values: 500 MB of disk space for the cache, with 50% of the cache space retained.

Example:

```python
config = NPURunConfig(op_compiler_cache_mode="enable")
```

## op_compiler_cache_dir

Disk cache directory for operator compilation.

The value can contain letters, digits, underscores (_), hyphens (-), and periods (.).

If the specified directory exists and is valid, the kernel_cache subdirectory is automatically created. If the specified directory does not exist but is valid, the system automatically creates this directory and the kernel_cache subdirectory.

The storage priority of operator compilation cache files is as follows:

op_compiler_cache_dir > ${ASCEND_CACHE_PATH}/kernel_cache > Default path ($HOME/atc_data)

For details about ASCEND_CACHE_PATH, see [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Example:

```python
config = NPURunConfig(op_compiler_cache_dir="/home/test/kernel_cache")
```

## aicore_num

Maximum number of Cube cores and Vector cores used for operator compilation.

Format: Integer 1|Integer 2, where the two values are separated by vertical bars (|). Integer 1 specifies the maximum number of Cube cores to use, and Integer 2 specifies the maximum number of Vector cores to use. Both values must be greater than 0 and less than or equal to the actual number of Cube cores and Vector cores available on the AI processor.

NOTE:

- This option is supported by the following products:
  - Atlas A3 products
  - Atlas A2 products
- The maximum number of Cube cores and Vector cores for different AI processors can be found in the CANN installation directory/<arch\>-linux/data/platform_config/<soc_version\>.ini file. The following example indicates that there are 24 Cube cores and 48 Vector cores on the AI processor.
  
  ```text
  [SoCInfo]
  ai_core_cnt=24
  cube_core_cnt=24
  vector_core_cnt=48
  ```

- In static shape scenarios, if an existing operator binary is reused during model compilation (that is, jit_compile is set to false), aicore_num does not take effect.

Example:

  ```python
  config = NPURunConfig(aicore_num="2|4")
  ```

## oo_constant_folding

Enables or disables constant folding.

Constant folding is to directly compute and replace the values of constant expressions during graph compilation, thereby reducing the memory usage. In most cases, you are advised to retain the default value to enable constant folding. However, some networks require more memory during compilation and running, and the constant memory is occupied throughout the lifecycle of the graph. If the total memory increases after constant folding, you can use this parameter to disable constant folding.

- True (default): enables constant folding.

  In this case, a node marked with the _grappler_do_not_remove attribute via TensorFlow's Grappler will not be folded, while other nodes that meet the folding conditions will still be folded.

- False: disables constant folding.

  ```python
  config = NPURunConfig(oo_constant_folding=True)
  ```

NOTE:

If constant folding is disabled and an error occurs during network compilation and running, an error message similar to the following will be displayed:

- Example 1:Error message from the debug log:

  ```text
  [ERROR] GE(3469659,python3.7):2025-02-25-05:** [ge_deleted_op.cc:21]3470503 Run: ErrorNo: 4294967295(failed) [Delete][Node] Node:HcomAllReduce/input type is ExpandDims, should be deleted by ge.
  ```

  This error indicates that the network contains an ExpandDims operator that requires constant folding during graph compilation, meaning that constant folding cannot be disabled.

- Example 2: Screen output with error code EZ3003:Error Message is :

  ```text
  Error Message is :
  EZ3003: [PID: 3482331] 2025-02-25-14:07:19.774.362 No supported Ops kernel and engine are found for [import/conv2d_1/convolutionimport/batch_normalization_1/FusedBatchNorm_1_filter_host], optype [ConvBnFilterHost].
  Possible Cause: The operator is not supported by the system. Therefore, no hit is found in any operator information library.
  ```

  This error indicates that the network contains a ConvBnFilterHost operator that requires constant folding during graph compilation, meaning that constant folding cannot be disabled.

Solution:

Enable constant folding by setting oo_constant_folding to True, and then use the _grappler_do_not_remove attribute via TensorFlow's Grappler to selectively disable constant folding for specific operators.
