# DumpConfig Constructor

## Description

Constructor of the  **DumpConfig**  class, which is used to configure the dump function.

## Prototype

```python
class DumpConfig():
    def __init__(self,
                 enable_dump=False,
                 dump_path=None,
                 dump_step=None,
                 dump_mode="output",
                 enable_dump_debug=False,
                 dump_debug_mode="all",
                 dump_data="tensor",
                 dump_layer=None)
```

## Parameters

- **enable_dump**: input, whether to enable the data dump function.

  - **True**: enabled. The dump file path is read from  **dump_path**.
  - **False**  \(default\): disabled.

    > [!NOTE]NOTE
    >
    >- Data dump and overflow/underflow data collection cannot be enabled at the same time. That is,  **enable_dump**  and  **enable_dump_debug**  cannot be set to  **True**  at the same time.
    >- If either  **enable_dump**  or  **enable_dump_debug**  is set to  **True**  and  **enable_exception_dump**  is set to  **1**  \(indicating that common ExceptionDump function is enabled\): For dynamic-shape networks, only  **enable_exception_dump**  takes effect. For static-shape networks,  **enable_exception_dump**  and either  **enable_dump**  or  **enable_dump_debug**  take effect.

- **dump_path**: input, path for storing dump files. Required when  **enable_dump**  or  **enable_dump_debug**  is set to  **True**.

    Create the specified path in advance in the environment \(either in a container or on the host\) where training is performed. The running user configured during installation must have the read and write permissions on this path. The path can be an absolute path or a path relative to the path where the training script is executed.

  - An absolute path starting with a slash \(/\), for example,  **/home/test/output**.
  - A relative path starting with a directory name, for example,  **output**.

- **dump_step**: input, iterations to dump.

    Separate multiple iterations using vertical bars \(|\), for example,  **0|5|10**. You can also use hyphens \(-\) to specify the iteration range, for example,  **0|3-5|10**.

    If this parameter is not set, dump data of all iterations is collected.

- **dump_mode**: input, dump mode, specifying whether the operator input or output is dumped. The options are as follows:

  - **input**: dumps only operator inputs.
  - **output**  \(default\): dumps only operator outputs.
  - **all**: dumps both operator inputs and outputs.

    > [!NOTE]NOTE
    > If this parameter is set to  **all**, the input data of some operators, such as collective communication operators HcomAllGather and HcomAllReduce, will be modified during execution. Therefore, the system dumps the operator input before operator execution and dumps the operator output after operator execution. In this way, the dumped input and output data of the same operator is flushed to disks separately, and multiple dump files are generated. After parsing the dump files, you can determine whether the data is an input or output based on the file content.

- **enable_dump_debug**: input, indicating whether to enable overflow/underflow data collection.

  - **True**: enables overflow/underflow data collection. The dump file path is read from  **dump_path**. An exception is reported if  **dump_path**  is set to  **None**.
  - **False**  \(default\): disables overflow/underflow data collection.

- **dump_debug_mode**: input, overflow/underflow detection mode. The options are as follows:
  - **aicore_overflow**: detects AI Core operator overflow, that is, detecting whether abnormal extreme values \(such as 65500, 38400, and 51200 in float16\) are output with normal inputs. Once such a fault is detected, analyze the cause of the overflow/underflow and modify the operator implementation based on the network requirements and operator logic.
  - **atomic_overflow**: detects Atomic Add overflow/underflow. Atomic Add overflow/underflow is detected when data is transferred from the UB to OUT after AI Core computation.
  - **all**: detects overflow/underflow of both AI Core operators and Atomic Add. Defaults to  **all**.

    > [!NOTE]NOTE
    > For Ascend 950PR&950DT products, Atlas A3 products and Atlas A2 products, only the default value  **all**  can be used.

- **dump_data**: input, specifying the type of operator content to be dumped. The options are as follows:

  - **tensor**  \(default\): dumps operator data.
  - **stats**: dumps operator statistics. The result file is in  **.csv**  format.

    In large-scale training scenarios, dumping a large amount of data takes a long time. You can dump the statistics of all operators, identify the operators that may be exceptions based on the statistics, and then dump the input or output data of these exception operators.

- **dump_layer**: input, specifying the operators to be dumped. Multiple operator names are separated by spaces. If this parameter is not set, all operators are dumped by default.

    If the input of the specified operator involves the data operator, the data operator information is also dumped.

## Returns

An object of the  **DumpConfig**  class, as an argument passed to the  **NPURunConfig**  call.

## Restrictions

**enable_dump**  and  **enable_dump_debug**  are mutually exclusive.

## Example

```python
from npu_bridge.npu_init import *
...
dump_config = DumpConfig(enable_dump=True, dump_path="/home/test/output", dump_step="0|5|10", dump_mode="all")
session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(dump_config=dump_config, session_config=session_config)
```
