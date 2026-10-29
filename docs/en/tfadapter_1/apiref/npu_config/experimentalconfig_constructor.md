# ExperimentalConfig Constructor

## Description

Constructs an object of the  **ExperimentalConfig**  class. This constructor is an extended option for debugging and may be changed in later versions. Therefore, it cannot be used in production environments.

## Prototype

```python
class ExperimentalConfig():
    def __init__(self,
                 graph_compiler_cache_dir=None,
                 ......
                 )
```

## Parameters

- **graph_compiler_cache_dir**: input, drive cache directory for graph compilation. If this parameter is not empty, the drive cache function for graph compilation takes effect.

    The graph compilation cache function supports drive persistence of graph compilation results. When graph compilation is performed again, the compilation results cached on the drive can be directly loaded to reduce the graph compilation duration.

    Note:

  - The configured cache directory must exist. Otherwise, the compilation fails.
  - During graph compilation, the cache file is determined based on the value of this parameter. If the cache file does not exist, the cache is saved. If the cache file exists, the existing cache is directly loaded.
  - After a graph is changed, the original cache file is unavailable. You need to manually delete the cache file from the cache directory or rebuild and generate a cache file.
  - The cache does not ensure cross-version compatibility. If the version is upgraded, clear the cache directory and rebuild and generate the cache.
  - This function does not support models with resource operators.

    Example:

    ```python
    graph_compiler_cache_dir="/root/build_cache_dir"
    ```

- **accelerate_train_mode**: input. If training takes more than one hour, you can trigger training acceleration to improve training performance by configuring this option.

    Based on the configured acceleration type, acceleration trigger mode, and the proportion of low-precision training processes, the software compiles and runs the corresponding proportion of training processes with reduced precision, while the remaining processes are compiled and run at their original precision.

    The value of this option is a string with three fields separated by vertical bars \(|\), for example,  **fast|step|0.9**.

  - The first field indicates the acceleration type, which can be  **fast**  or  **fast1**.
    - **fast**: that the compilation is performed based on the float16 data type during precision reduction.
    - **fast1**: that the compilation is performed based on the bf16 data type during precision reduction.

  - The second field supports two values:  **step**  and  **loss**, indicating that the entire training process is divided into low-precision training and high-precision training based on the  **step**  or  **loss**  value, respectively.
  - The third field indicates the proportion of the training process that runs in low precision, relative to either the total step count or the total loss range.
    - When the second field is  **step**, its value ranges from 0.2 to 0.9. The default value is  **0.9**.
    - When the second field is  **loss**, its value ranges from 1.01 to 1.5. The default value is  **1.05**.

    Example:

  - Acceleration triggered by  **step**:

    ```python
    accelerate_train_mode="fast|step|0.9"
    ```

  - Acceleration triggered by  **loss**:

     ```python
    accelerate_train_mode="fast|loss|1.05"
     ```

    **Note:**

    1. To use this option for training acceleration, ensure that the network script can converge properly.
    2. For training scripts with short execution time, enabling this option may not bring positive end-to-end performance gains.
    3. The function of this option is related to the precision mode configured in the network script:
        - When  **precision_mode**  is used to configure the precision mode, this option can be enabled only when  **precision_mode**  is set to  **allow_fp32_to_fp16**,  **must_keep_origin_dtype**, or  **none**.
        - When  **precision_mode_v2**  is used to configure the precision mode, this option can be enabled only when  **precision_mode_v2**  is set to  **origin**  or  **none**.

    4. This option is related to the number of small loops. Enabling small loops may result in inaccurate splitting of the training process based on the specified steps or  **loss**  value, which may ultimately affect the loss and accuracy.
    5. When this option is enabled, you need to modify the network script and use  [TellMeStepOrLossHook Constructor](../npu_hook/TellMeStepOrLossHook_constructor.md)  to notify the bottom-layer software of the serial number of the current  **step**  and the total number of  **step**s, or the current  **loss**  and the target  **loss**.

        Example:

        ```python
        from npu_bridge.npu_init import *
        from npu_bridge.estimator.npu.npu_config import ExperimentalConfig
        from npu_bridge.estimator.npu.npu_hook import TellMeStepOrLossHook
        # Enable the fast acceleration mode. The training process is divided based on the ratio of 90% to the total steps. That is, low-precision training is performed on 90% of the total steps, and high-precision training is performed on the remaining steps.
        experimental_config = ExperimentalConfig(accelerate_train_mode="fast|step|0.9")
        config = NPURunConfig(experimental_config=experimental_config)
        est = NPUEstimator(
        model_fn=model_fn,
        config=config,
        params=params)
        hooks = []
        max_steps = 10000
        # step splitting mode, which notifies the bottom-layer software of the serial number of the current step and the total number of steps. The value global_step:0 is only an example. Set it to the actual tensor name of the current step.
        my_hook = TellMeStepOrLossHook(step='global_step:0', total_step=max_steps )
        # loss splitting mode, which notifies the bottom-layer software of the current loss and the target loss. The value loss:0 is only an example. Set it to the actual tensor name of the current loss.
        # my_hook = TellMeStepOrLossHook(loss='loss:0', final_loss=7.1)
        hooks.append(my_hook)
        # Start training.
        est.train(
        input_fn=imagenet_train.input_fn,
        max_steps=max_steps,
        hooks=hooks)
        ```

## Returns

An object of the  **ExperimentalConfig**  class, as an argument passed to the  **NPURunConfig**  call.

## Restrictions

None

## Example

```python
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu.npu_config import ExperimentalConfig
...
experimental_config=ExperimentalConfig(accelerate_train_mode="fast|step|0.9")
session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(experimental_config=experimental_config, session_config=session_config)
```
