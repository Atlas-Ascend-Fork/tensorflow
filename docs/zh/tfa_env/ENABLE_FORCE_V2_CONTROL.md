# ENABLE_FORCE_V2_CONTROL

## 功能描述

TensorFlow 1.15训练场景下，如果输入是动态shape，由于tf.case/tf.cond/tf.while_loop这些API对应TensorFlow V1版本的控制流算子（例如Switch、Merge、Enter、LoopCond、NextIteration、Exit、ControlTrigger等）不支持动态shape，仅TensorFlow V2版本的控制流算子（例如If、Case、While、For、PartitionedCall等）支持动态shape，因此，如果用户的训练脚本中使用了这些API，需要将V1版本的控制流算子转换为V2版本，用于支持动态shape功能。另外，如果网络中的分支结构较多，采用V1版本的控制流算子可能导致流数超限，此时也需要将V1版本的控制流算子转换成V2版本算子解决。

此环境变量设置为“1”时，代表开启V1版本的控制流算子转换成V2版本算子的功能，设置为其他值时不会开启此功能。

## 配置示例

```bash
export ENABLE_FORCE_V2_CONTROL=1
```

## 使用约束

- 该环境变量仅适用于TensorFlow  1.15训练场景。
- 使用该环境变量，**可能会出现V1版本控制流算子到V2版本控制流算子转换失败的情况**，例如网络脚本中带ref控制算子的场景。

    为避免V1到V2控制流算子转换失败，**建议通过修改网络脚本的方式**，实现V1版本控制流算子到V2版本控制流算子的转换。

    在import tensorflow as tf后增加如下两条指令：

    ```python
    tf.enable_control_flow_v2()
    tf.enable_resource_variables()
    ```

- 此环境变量不支持通过底层ops调用控制类算子场景的转换，例如tf.raw_ops.Merge 、tf.raw_ops.Switch，请手工修改网络脚本。
- 使用该环境变量将V1版本控制流算子转换到V2版本控制流算子后，可能存在性能下降的情况。

## 支持的型号

Ascend 950PR&950DT系列产品

Atlas A3系列产品

Atlas A2系列产品

Atlas训练系列产品
