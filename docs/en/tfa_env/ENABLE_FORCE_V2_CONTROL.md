# ENABLE_FORCE_V2_CONTROL

## Description

In the TensorFlow 1.15 training scenario, if the input has a dynamic shape, upgrade the control flow operators of the V1 version to those of the V2 version to support the dynamic shape function. Only TensorFlow V2 control flow operators \(such as If, Case, While, For, and PartitionedCall\) support dynamic shapes. TensorFlow V1 control flow operators \(such as Switch, Merge, Enter, LoopCond, NextIteration, Exit, and ControlTrigger\) corresponding to the  **tf.case**,  **tf.cond**, and  **tf.while_loop**  APIs do not support dynamic shapes. If the network has many branch structures, upgrade the control flow operators of the V1 version to those of the V2 version. Otherwise, the flow of data may exceed the limit.

If this environment variable is set to  **1**, the function of converting V1 control flow operators to V2 control flow operators is enabled. If this environment variable is set to other values, this function is disabled.

## Example

```bash
export ENABLE_FORCE_V2_CONTROL=1
```

## Constraints

- This environment variable applies only to the  TensorFlow  1.15 training scenario.
- If this environment variable is used, the control flow operator of the V1 version may fail to be converted to the control flow operator of the V2 version, for example, the ref control operator contained in the network script.

    **You are advised to modify the network script**  to convert control flow operators of the V1 version to control flow operators of the V2 version.

    Add the following two instructions after  **import tensorflow as tf**:

    ```python
    tf.enable_control_flow_v2()
    tf.enable_resource_variables()
    ```

- This environment variable does not support conversion in scenarios where control operators are called through underlying ops, for example, tf.raw_ops.Merge or tf.raw_ops.Switch. You need to manually modify the network script.
- After this environment variable is used to convert V1 control flow operators to V2 control flow operators, the performance may deteriorate.

## Applicability

Ascend 950PR&950DT products

Atlas A3 products

Atlas A2 products

Atlas training products
