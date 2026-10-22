# Debugging

## op_debug_config

Enable for global memory check.

The value is the path of the .cfg configuration file. Multiple options in the configuration file are separated by commas (,).

- oom: checks whether memory overwriting occurs in the global memory during operator execution.

  During operator compilation, the .o file (operator binary file) and .json file (operator description file) are retained in the kernel_meta folder in the current execution path, and the following detection logic is added:

  ```c
  inline __aicore__ void  CheckInvalidAccessOfDDR(xxx) {
      if (access_offset < 0 || access_offset + access_extent > ddr_size) {
          if (read_or_write == 1) {
              trap(0X5A5A0001);
          } else {
              trap(0X5A5A0002);
          }
      }
  }
  ```

  You can use dump_cce to view the preceding code in the generated .cce file.

  If memory overwriting occurs during compilation, the error code EZ9999 is reported.

- dump_cce: retains the .cce file, .o file, and .json file of the operator in the kernel_meta folder in the current execution path during operator compilation.
- dump_loc: retains the .cce file, .o file, and .json file of the operator, as well as the \_loc.json file (mapping file of python-cce) in the kernel_meta folder in the current execution path during operator compilation.
- ccec_O0: enables the default option -O0 of the CCEC during operator compilation. This option does not perform any optimization based on the debugging information.
- ccec_g: enables the -g option of the CCEC during operator compilation. Compared with -O0, this option generates optimization and debugging information.
- check_flag: checks whether pipeline synchronization signals in operators match each other during operator execution.

Retain the .o file and .json file in the generated kernel_meta folder and add the following detection logic during operator compilation:

```text
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  ....
  pipe_barrier(PIPE_MTE3);
  pipe_barrier(PIPE_MTE2);
  pipe_barrier(PIPE_M);
  pipe_barrier(PIPE_V);
  pipe_barrier(PIPE_MTE1);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  ...
```

You can use dump_cce to view the preceding code in the generated .cce file.

During compilation, if a mismatch exists in the pipeline synchronization signals in an operator, a timeout error is reported at the faulty operator. The following is an example of the error message:

```text
Aicore kernel execute failed, ..., fault kernel_name=operator name,...
rtStreamSynchronizeWithTimeout execute failed....
```

Example:

```python
npu.global_options().op_debug_config="/root/test0.cfg"
```

The information about the test0.cfg file is as follows:

```python
op_debug_config=ccec_g,oom
```

Constraints:

During operator compilation, if you want to compile only some instead of all AI Core operators, add the op_debug_list field to the test0.cfg configuration file. By doing so, only the operators specified in the list are built, based on the options configured in op_debug_config. The op_debug_list field has the following requirements:

- The operator name or operator type can be specified.
- Operators are separated by commas (,). The operator type is configured in the OpType::typeName format. The operator type and operator name can be configured in a mixed manner.
- The operator to be compiled must be stored in the configuration file specified by op_debug_config.

The following is a configuration example of the test0.cfg file:

```text
op_debug_config= ccec_g,oom
op_debug_list=GatherV2,opType::ReduceSum
```

During model compilation, the GatherV2 and ReduceSum operators are compiled based on the ccec_g and oom options.

When ccec_O0 and ccec_g are enabled, the size of the operator kernel file (*.o file) increases. In dynamic shape scenarios, all possible scenarios are traversed during operator compilation, which may cause operator compilation failures due to large operator kernel files. In this case, do not enable the CCEC options.
If the compilation failure is caused by large operator kernel files, the following log is displayed:

```text
message:link error ld.lld: error: InputSection too large for range extension thunk ./kernel_meta_xxxxx.o:(xxxx)
```

- The CCEC options ccec_O0 and oom cannot be enabled at the same time. Otherwise, an AI Core error is reported. The following is an example of the error message:

```text
...there is an aivec error exception, core id is 49, error code = 0x4 ...
```

- If this parameter is set to dump_cce or dump_loc, you can use debug_dir to specify the path for storing debugging-related process files.
- When the build options oom, dump_cce, and dump_loc are configured, if the model contains the following MC2 operators, the *.o, *.json, and *.cce files of the operators are not generated in the operator build folder kernel_meta.

  ```text
  MatMulAllReduce
  MatMulAllReduceAddRmsNorm
  AllGatherMatMul
  MatMulReduceScatter
  AlltoAllAllGatherBatchMatMul
  BatchMatMulReduceScatterAlltoAll
  ```

- If NPU_COLLECT_PATH is configured, the function of checking whether memory overwriting occurs in the global memory cannot be enabled. That is, the configuration file specified by this parameter cannot be set to oom. Otherwise, an error is reported when the compiled model file or operator kernel package is used.

## enable_exception_dump

Whether to dump data of exception operators.

- 0: disables the exception operator data dump function.
- 1: enables the common ExceptionDump function to dump the input and output data, tensor description information (such as shape, dtype, and format), and workspace information of exception operators.
  
  The dump data is stored in the following directories in descending order of priority: NPU_COLLECT_PATH > ASCEND_WORK_PATH > default directory (extra-info in the script execution directory).

- 2 (default): enables the LiteExceptionDump function to dump the input and output data, workspace information, and tiling information of exception operators. The exported data is used to analyze AI Core errors. For details about how to collect and locate AI Core errors, see "Typical Faults > AI Core Error Locating" in [Troubleshooting](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/troubleshooting/troubleshooting_0001.html).

  The dump data is stored in the following directories in descending order of priority: ASCEND_WORK_PATH > default directory (extra-info/data-dump/<device_id\> in the script execution directory).

> [!NOTE]NOTE
>If the environment variable NPU_COLLECT_PATH is configured, exception operator data is dumped in accordance with mode 1 (common ExceptionDump) regardless of the value of enable_exception_dump, and the dump data is stored in the directory specified by NPU_COLLECT_PATH.
>For details about environment variables, see [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Example:

```python
npu.global_options().enable_exception_dump=1
```

## debug_dir

Directory of the debug files generated during operator building, including the .o, .json, and .cce files.

The storage priority of the debugging files generated during operator compilation is as follows:

debug_dir > ASCEND_WORK_PATH > Default storage path (current script execution path)

Example:

```python
npu.global_options().debug_dir="/home/test"
```

## export_compile_stat

Whether to generate the operator fusion result file fusion_result.json during graph compilation. 

The values are as follows:

- 0: The operator fusion result file is not generated.
- 1 (default): The operator fusion result file is generated when the program exits normally.
- 2: The operator fusion result file is generated after graph compilation is complete. That is, if graph compilation is complete but the program is interrupted, the result file is also generated.

The fusion_result.json file records the fusion patterns used during graph compilation. The key fields in the file are described as follows:

- session_and_graph_id_xx_xx: thread and graph ID of the fusion result.
- graph_fusion: graph fusion.
- ub_fusion: UB fusion. The Ascend 950PR&950DT products do not support UB fusion, and therefore this information is not generated.

  **The Ascend 950PR&950DT products do not support UB fusion and therefore does not generate this information**.

- match_times: number of times that the fusion pattern is matched during graph build.
- effect_times: actual number of times that the fusion takes effect.
- repository_hit_times: number of times that the UB fusion repository is hit. The Ascend 950PR&950DT products do not support UB fusion, and therefore this information is not generated.

  **The Ascend 950PR&950DT products do not support UB fusion and therefore does not generate this information**.

NOTE:

- If ASCEND_WORK_PATH is not configured in the environment, the operator fusion result is saved to the fusion_result.json file in the current execution directory. If ASCEND_WORK_PATH is configured, the operator fusion result is saved to the $ASCEND_WORK_PATH/FE/${Process ID}/fusion_result.json file. For details about environment variables, see [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).
- The fusion patterns disabled by fusion_switch_file are not displayed in fusion_result.json.

Example:

```python
npu.global_options().export_compile_stat=1
```
