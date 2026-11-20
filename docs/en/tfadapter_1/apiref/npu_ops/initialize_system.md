# initialize_system

## Description

Excludes the GE initialization time in the training time statistics. Generally, this API is not required for training. Before using the collective communication API, call this API to initialize the collective communication.

## Prototype

```python
def initialize_system(name = None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| name | Input | Operator name. |

## Returns

An operator for the user to initialize GE by using `sess.run(op)`.

## Restrictions

If the  **initialize_system**  API needs to be called and the following functions need to be enabled during training, the configuration must be performed when a session is started in  **initialize_system**.

- **profiling_mode**: whether to enable profiling.
  - **True**: enabled. The profiling options are determined by  **profiling_options**.
  - **False**  \(default\): disabled.

- **profiling_options**: profiling options.

    For supported options, refer to the environment variable [PROFILING_OPTIONS](https://gitcode.com/cann/oam-tools/blob/master/docs/en/user_guide/env-vars/PROFILING_OPTIONS.md).

- **enable_dump**: whether to enable the data dump function.

  - **True**: enabled. The dump file path is read from  **dump_path**. If  **dump_path**  is set to  **None**, an exception occurs.
  - **False**  \(default\): disabled.

    > [!NOTE]NOTE
    >
    >- Data dump and overflow/underflow data collection cannot be enabled at the same time. That is,  **enable_dump**  and  **enable_dump_debug**  cannot be set to  **True**  at the same time.
    >- If either  **enable_dump**  or  **enable_dump_debug**  is set to  **True**  and  **enable_exception_dump**  is set to  **1**  \(indicating that common ExceptionDump function is enabled\): For dynamic-shape networks, only  **enable_exception_dump**  takes effect. For static-shape networks,  **enable_exception_dump**  and either  **enable_dump**  or  **enable_dump_debug**  take effect.

- **dump_path**: path for storing dump files. Required when  **enable_dump**  or  **enable_dump_debug**  is set to  **True**.

    Create the specified path in advance in the environment \(either in a container or on the host\) where training is performed. The running user configured during installation must have the read and write permissions on this path. The path can be an absolute path or a path relative to the path where the training script is executed.

  - An absolute path starting with a slash \(/\), for example,  **/home/test/output**.
  - A relative path starting with a directory name, for example,  **output**.

- **dump_step**: iterations to dump. Defaults to  **None**, indicating that all iterations are dumped.

    Separate multiple iterations using vertical bars \(|\), for example,  **0|5|10**. You can also use hyphens \(-\) to specify the iteration range, for example,  **0|3-5|10**.

- **dump_mode**: dump mode, specifying whether the operator input or output is dumped. The options are as follows:

  - **input**: dumps only operator inputs.
  - **output**  \(default\): dumps only operator outputs.
  - **all**: dumps both operator inputs and outputs.

    > [!NOTE]NOTE
    >If this parameter is set to  **all**, the input data of some operators, such as collective communication operators HcomAllGather and HcomAllReduce, will be modified during execution. Therefore, the system dumps the operator input before operator execution and dumps the operator output after operator execution. In this way, the dumped input and output data of the same operator is flushed to disks separately, and multiple dump files are generated. After parsing the dump files, you can determine whether the data is an input or output based on the file content.

- **enable_dump_debug**: indicates whether to enable overflow/underflow detection.
  - **True**: enabled. The dump file path is read from  **dump_path**. If  **dump_path**  is set to  **None**, an exception occurs.
  - **False**  \(default\): disabled.

    > [!NOTE]NOTE
    >
    > - Data dump and overflow/underflow data collection cannot be enabled at the same time. That is,  **enable_dump**  and  **enable_dump_debug**  cannot be set to  **True**  at the same time.
    > - If either  **enable_dump**  or  **enable_dump_debug**  is set to  **True**  and  **enable_exception_dump**  is set to  **1**  \(indicating that common ExceptionDump function is enabled\): For dynamic-shape networks, only  **enable_exception_dump**  takes effect. For static-shape networks,  **enable_exception_dump**  and either  **enable_dump**  or  **enable_dump_debug**  take effect.

- **dump_debug_mode**: overflow/underflow detection mode. The values are as follows:
  - **aicore_overflow**: detects AI Core operator overflow/underflow, that is, detecting whether abnormal extreme values \(such as 65500, 38400, and 51200 in float16\) are output with normal inputs. Once such a fault is detected, analyze the cause of the overflow/underflow and modify the operator implementation based on the network requirements and operator logic.
  - **atomic_overflow**: detects Atomic Add overflow/underflow. Atomic Add overflow/underflow is detected when data is transferred from the UB to OUT after AI Core computation.
  - **all**: detects overflow/underflow of both AI Core operators and Atomic Add. The default value is  **all**.

    > [!NOTE]NOTE
    >For the  Ascend 950PR&950DT products,  Atlas A3 products, and  Atlas A2 products, only the default value  **all**  can be used.

- **precision_mode**: operator precision mode, which must be of the string type.

  - **allow_fp32_to_fp16**
    - For matrix operators:
      - If the operator precision in the original graph is float32, the precision is preferably reduced to float16. If the operator in the AI Core does not support float16, float32 is used. If the operator in the AI Core does not support float32, the AI CPU operator is used for computation. If the AI CPU operator also does not support float32, an error is reported during execution.
      - If the operator precision in the original graph is bfloat16, the precision of the original graph is preferably used. If the operator in the AI Core does not support bfloat16, float32 is used. If the operator in the AI Core does not support float32, the precision is directly reduced to float16. If the operator in the AI Core does not support float16, the AI CPU operator is used for computation. If the AI CPU operator also does not support float16, an error is reported during execution.

    - For vector operators, the precision of the original graph is retained preferably.
      - If the operator precision in the original graph is float32, the precision of the original graph is preferably used. If the operator in the AI Core does not support float32, the precision is directly reduced to float16. If the operator in the AI Core does not support float16, the AI CPU operator is used for computation. If the AI CPU operator also does not support float16, an error is reported during execution.
      - If the operator precision in the original graph is bfloat16, the precision of the original graph is preferably used. If the operator in the AI Core does not support bfloat16, float32 is used. If the operator in the AI Core does not support float32, the precision is directly reduced to float16. If the operator in the AI Core does not support float16, the AI CPU operator is used for computation. If the AI CPU operator also does not support float16, an error is reported during execution.

  - **force_fp16**

    Forces float16 for operators supporting float16, bfloat16, and float32. This parameter applies only to online inference scenarios.

  - **force_fp32/cube_fp16in_fp32out**

    **force_fp32**  and  **cube_fp16in_fp32out**  have the same effect. This option indicates that the system selects different processing modes based on the operator type when the operator in the AI Core supports both the float32 and float16 data types.  **cube_fp16in_fp32out**  is newly added to the new version. For cube operators, this option has clearer semantics.

    - For cube operators, the system processes the computation based on the operator implementation.
        1. The preferred input data type is float16 and the output data type is float32.
        2. If the float16 input data and float32 output data types are not supported, set both the input and output data types to float32.
        3. If the float32 input and output data types are not supported, set both the input and output data types to float16.
        4. If the float16 input and output data types are not supported, an error is reported.

    - For vector compute operators, the operator precision in the original graph is float16 or bfloat16, and float32 is forcibly selected.

      This option is invalid if the original graph contains operators not supporting float32 in the AI Core, for example, an operator that supports only float16. In this case, float16 is retained. If the operator in the AI Core does not support float32 and it is configured to the blocklist of precision reduction \(by setting  **precision_reduce**  to  **false**\), the counterpart AI CPU operator supporting float32 is used. If the AI CPU operator does not support float32, an error is reported.

  - **must_keep_origin_dtype**

    Retains the original precision.

    - If the precision of an operator in the original graph is float16, and the implementation of the operator in the AI Core does not support float16 but supports only float32 and bfloat16, the system automatically uses high-precision float32.
    - If the precision of an operator in the original graph is float16, and the implementation of the operator in the AI Core does not support float16 but supports only bfloat16, the AI CPU operator of float16 is used. If the AI CPU operator is not supported, an error is reported.
    - If the precision of an operator in the original graph is float32, and the implementation of the operator in the AI Core does not support float32 but supports only float16, the AI CPU operator of float32 is used. If the AI CPU operator is not supported, an error is reported.

  - **allow_mix_precision_fp16/allow_mix_precision**

    **allow_mix_precision**  has the same effect as that of  **allow_mix_precision_fp16**, indicating that mixed precision of float16, bfloat16, and float32 is used for neural network processing.  **allow_mix_precision_fp16**  is newly added to the new version, which has clearer semantics for easy understanding.

    For float32 and bfloat16 operators in the original model, float16 is automatically used for certain float32 and bfloat16 operators based on the built-in tuning policy. This will improve system performance and reduce memory usage with minimal precision degradation.

  - **allow_mix_precision_bf16**

    Mixed precision of bfloat16 and float32 is used for neural network processing. In this mode, bfloat16 is automatically used for certain float32 operators on the original model based on the built-in tuning policy. This will improve system performance and reduce memory usage with minimal precision degradation. If the operator in the AI Core does not support bfloat16 and float32, the AI CPU operator is used for computation. If AI CPU operator also does not support bfloat16 and float32, an error is reported during execution.

    Note: This configuration is supported only by the  Ascend 950PR&950DT products,  Atlas A3 products, and  Atlas A2 products.

  - **allow_fp32_to_bf16**

    - If the operator precision in the original graph is float32, the precision of the original graph is preferably used. If the operator in the AI Core does not support float32, the precision is reduced to bfloat16. If the operator in the AI Core does not support bfloat16, the AI CPU operator is used for computation. If the AI CPU operator also does not support bfloat16, an error is reported during execution.
    - If the operator precision in the original graph is bfloat16, the precision of the original graph is preferably used. If the operator in the AI Core does not support bfloat16, float32 is used. If the operator in the AI Core does not support float32, the AI CPU operator is used for computation. If the AI CPU operator also does not support float32, an error is reported during execution.

    Note: This configuration is supported by the  Ascend 950PR&950DT products,  Atlas A3 products, and  Atlas A2 products.

    For the  Atlas training products, the default value is  **allow_fp32_to_fp16**.

    For the  Atlas A2 products, the default value is  **must_keep_origin_dtype**.

- **graph_run_mode**: graph run mode The values are as follows:
  - **0**: online inference.
  - **1**  \(default\): training.

- **op_debug_level**: indicates whether to enable operator debugging. The values are as follows:
  - **0**: disables operator debug.
  - **1**: Enables operator debug. TBE instruction mapping files are generated in the  **kernel_meta**  directory under the training script execution path, including operator CCE files \(.cce\), Python-CCE mapping files \(_loc.json\), .o files, and .json files. These files are used for AI Core error analysis with related tools.

    Note: For the  Ascend 950PR&950DT products, no TBE instruction mapping files are generated.

  - **2**: Enables operator debug. TBE instruction mapping files are generated in the  **kernel_meta**  directory under the training script execution path, including operator CCE files \(.cce\), Python-CCE mapping files \(_loc.json\), .o files, and .json files. The compilation optimization of the CCE compiler is disabled and the CCE compiler debugging function is enabled \(by setting the compiler option to  **-O0-g**\). These files are used for AI Core error analysis with related tools.

    Note: For the  Ascend 950PR&950DT products, no TBE instruction mapping files are generated.

  - **3**: disables operator debug. The operator .o and .json files are retained in the  **kernel_meta**  folder in the training script execution directory.
  - **4**: disables operator debug. The operator binary \(.o\) and operator description file \(.json\) are retained, and a TBE instruction mapping file \(.cce\) and a UB fusion description file \(**_\{$kernel_name\}__compute.json**\) are generated in the  **kernel_meta**  folder under the training script execution directory.

    Note: For the  Ascend 950PR&950DT products, neither TBE instruction mapping files nor UB fusion compute description files are generated.

    NOTICE:

    - If this option is set to  **0**  and  **op_debug_config**  is configured, the operator compilation directory  **kernel_meta**  is still generated in the current execution path during training. The content generated in the directory is subject to  **op_debug_config**.
    - You are advised to set this option to  **0**  or  **3**  for training. To locate AI Core errors, set this parameter to  **1**  or  **2**, which might compromise the network performance.
    - If this option is set to  **2**  \(the CCE compiler is enabled\), it cannot be used together with the  **oom**  option in  **op_debug_config**. Otherwise, an AI Core error is reported. The following is an example of the error message:

       ```text
       ...there is an aivec error exception, core id is 49, error code = 0x4 ...
       ```

    - If this parameter is set to  **2**  \(the CCE compiler is enabled\), the size of the operator kernel file \(\*.o file\) increases. In dynamic shape scenarios, all possible scenarios are traversed during operator build, which may cause operator build failures due to large operator kernel files. In this case,  **2**  is not recommended.
        If the build failure is caused by the large operator kernel file, the following log is displayed:

        ```text
        message:link error ld.lld: error: InputSection too large for range extension thunk ./kernel_meta_xxxxx.o:(xxxx)
        ```

    - If the value of this parameter is not  **0**, you can use the  **debug_dir**  parameter to specify the path for storing debugging-related process files.
    - If this parameter is set to  **0**  and  **NPU_COLLECT_PATH**  is set, the operator compilation directory  **kernel_meta**  is generated in the current path after the command is executed. If  **ASCEND_WORK_PATH**  is set,  **kernel_meta**  is generated in the path specified by the environment variable. For details about environment variables, see  [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).
    - When the debug function is enabled, if the model contains the following merged compute and communication \(MC2\) operators, the  **\*.o**,  **\*.json**, and  **\*.cce**  files of the operators are not generated in the operator build folder  **kernel_meta**.

    ```text
    MatMulAllReduce
    MatMulAllReduceAddRmsNorm
    AllGatherMatMul
    MatMulReduceScatter
    AlltoAllAllGatherBatchMatMul
    BatchMatMulReduceScatterAlltoAll
    ```

- **enable_exception_dump**: indicates whether to dump data of exception operators.

  - **0**: disables the exception operator data dump function.
  - **1**: enables the common ExceptionDump function to dump the input and output data, tensor description information \(such as shape, dtype, and format\), and workspace information of exception operators.

    The dump data is stored in the following directories in descending order of priority:  **NPU_COLLECT_PATH**  \>  **ASCEND_WORK_PATH**  \> default directory \(**extra-info**  in the script execution directory\).

  - **2**  \(default\): enables the LiteExceptionDump function to dump the input and output data, workspace information, and tiling information of exception operators. The exported data is used to analyze AI Core errors.

    The dump data is stored in the following directories in descending order of priority:  **ASCEND_WORK_PATH**  \> default directory \(**extra-info/data-dump/<**_device_id_**\>**  in the script execution directory\).

    > [!NOTE]NOTE
    >If the environment variable  **NPU_COLLECT_PATH**  is configured, exception operator data is dumped in accordance with mode 1 \(common ExceptionDump\) regardless of the value of  **enable_exception_dump**, and the dump data is stored in the directory specified by  **NPU_COLLECT_PATH**.

- **op_select_implmode**: operator implementation mode. Certain operators built in the NPU can be implemented in either high-precision or high-performance mode at model build time. Arguments:
  - **high_precision**: high-precision implementation mode. In high-precision mode, Taylor's theorem or Newton's method is used to improve operator precision with float16 input.
  - **high_performance**  \(default\): high-performance implementation mode. In high-performance mode, the optimal performance is implemented without affecting the network precision \(float16\).

- **optypelist_for_implmode**: list of operator types \(separated by commas\) that use the mode specified by the  **op_select_implmode**  parameter. Currently, Pooling, SoftmaxV2, LRN, and ROIAlign operators are supported.

    Use this parameter in conjunction with  **op_select_implmode**, for example:

    Set  **op_select_implmode**  to  **high_precision**.

    Set  **optypelist_for_implmode**  to  **Pooling**.

    This parameter is left empty by default, indicating that the configuration is disabled.

## Example

If you use an HCCL API such as  **get_local_rank_id**,  **get_rank_size**, or  **get_rank_id**  before  **sess.run\(\)**  or  **estimator.train\(\)**, you need to start another session and execute  **initialize_system**  to initialize collective communication. After the training is complete, execute  **shutdown_system**  and close the session.

```python
import tensorflow as tf
from npu_bridge.npu_init import *

npu_int = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

init_sess = tf.Session(config=config)
init_sess.run(npu_int)

# Call an HCCL API...
# Perform training...

init_sess.run(npu_shutdown)
init_sess.close()
```

Or:

```python
import tensorflow as tf
from npu_bridge.npu_init import *

npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
    sess.run(npu_init)
    # Call an HCCL API...
    # Perform training...
    sess.run(npu_shutdown)
```
