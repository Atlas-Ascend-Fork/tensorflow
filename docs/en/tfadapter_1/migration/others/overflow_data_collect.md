# Overflow/Underflow Data Collection

## Overview

For larger networks, a large amount of data will be dumped during operator accuracy analysis. In addition, due to the randomness of the network, it is difficult to locate the operators with accuracy problems by comparing with the benchmark data. In this case, you can choose to enable overflow/underflow data collection. Currently, the following three overflow/underflow detection modes are provided:

- **aicore_overflow**: detects AI Core operator overflow/underflow, that is, detecting whether abnormal extreme values \(such as 65500, 38400, and 51200 in float16\) are output with normal inputs. Once such a fault is detected, analyze the cause of the overflow/underflow and modify the operator implementation based on the network requirements and operator logic.
- **atomic_overflow**: detects Atomic Add overflow/underflow. Atomic Add overflow/underflow is detected when data is transferred from the UB to OUT after AI Core computation.
- **all**: detects overflow/underflow of both AI Core operators and Atomic Add.

Based on the overflow/underflow detection result, locate the faulty operators, dump data of the specific faulty operators, and analyze the dump data to solve the accuracy drop issue.

> [!NOTE]NOTE
>
> - For  Ascend 950PR&950DT products,  Atlas A3 products,  Atlas A2 products, the overflow/underflow detection mode can only be  **all**.
> - If you need to enable overflow/underflow data collection \(disabled by default\), modify the training script as described in this section. A simpler method is provided in  [Floating-Point Exception Detection](../accuracy_debugging/floating-point_exception_detection.md).

## Precautions

- Data dump \(**enable_dump**\) is mutually exclusive with overflow/underflow detection \(**enable_dump_debug**\).
- The overflow/underflow data collection function or data dump function might cause a full disk due to the generation of result files. You are advised to limit the number of iterations appropriately.

## In Estimator Mode

In  **Estimator**  mode, use  **dump_config**  in  **NPURunConfig**  to set the overflow/underflow detection mode. Before creating  **NPURunConfig**, instantiate an instance of class  **DumpConfig**. For details about  **DumpConfig**, see the corresponding API description.

```python
from npu_bridge.npu_init import *

# dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
# enable_dump_debug: whether to collect overflow/underflow data.
# dump_debug_mode: overflow/underflow detection mode select, which can be all, aicore_overflow, or atomic_overflow
dump_config = DumpConfig(enable_dump_debug = True, dump_path = "/home/test/output", dump_debug_mode = "all" )
session_config=tf.ConfigProto()

config = NPURunConfig(
    dump_config=dump_config,
    session_config=session_config)
```

## In sess.run Mode

In  **sess.run**  mode, set the overflow/underflow detection mode by setting the session configuration options  **dump_path**,  **enable_dump_debug**, and  **dump_debug_mode**.

```python
config = tf.ConfigProto()

custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True

# dump_path: dump path. Create the specified path in advance in the training environment (either in a container or on the host). The running user configured during installation must have the read and write permissions on this path.
custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/test/output")
# enable_dump_debug: whether to collect overflow/underflow data.
custom_op.parameter_map["enable_dump_debug"].b = True
# dump_debug_mode: overflow/underflow detection mode select, which can be all, aicore_overflow, or atomic_overflow
custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF


with tf.Session(config=config) as sess:
  print(sess.run(cost))
```

## Viewing and Parsing Overflow/Underflow Data

By default, the generated overflow/underflow operator data file is stored in the  **_\{dump_path\}/\{time\}/\{device_id\}/\{model_name\}/\{model_id\}/\{data_index\}_**  directory, for example,  **/home/HwHiAiUser/output/20200808163566/0/npu_cluster_0/11/0**. If no overflow/underflow data is collected, that is, no overflow occurs, the preceding directory is not generated.

For details about overflow/underflow data files and how to parse them, see  Extended Functions \> Overflow/Underflow Operator Data Collection and Analysis  in  [Accuracy Analyzer](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/ModelAccuracyAnalyzer/atlasaccuracy_16_0001.html).
