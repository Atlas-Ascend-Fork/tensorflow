# ExperimentalConfig构造函数

## 功能说明

ExperimentalConfig类的构造函数，调试功能扩展参数，后续版本可能会存在变动，不支持应用于生产环境。

## 函数原型

```python
class ExperimentalConfig():
    def __init__(self,
                 graph_compiler_cache_dir=None,
                 ……
                 )
```

## 参数说明

- **graph_compiler_cache_dir**：输入，该参数用于配置图编译磁盘缓存目录，当该参数配置为非空时，图编译磁盘缓存功能生效。

    图编译缓存功能支持将图编译结果进行磁盘持久化，当再次执行图编译运行时，可直接加载磁盘上缓存的编译结果，从而减少图编译时长。

    需要注意：

  - 配置的缓存目录必须存在，否则会导致编译失败。
  - 图编译时，会根据此参数的值确定缓存文件，若缓存文件不存在则保存缓存，若缓存文件存在则直接加载缓存。
  - 图发生变化后，原来的缓存文件不可用，用户需要手动删除缓存目录中的缓存文件，然后重新编译生成缓存文件。
  - 缓存不保证跨版本的兼容性，如果版本升级，需要清理缓存目录重新编译生成缓存。
  - 该功能当前不支持带资源类算子的模型。

    配置示例：

    ```python
    graph_compiler_cache_dir="/root/build_cache_dir"
    ```

- **accelerate_train_mode**：输入，针对超过1小时以上的训练场景，开发者可以通过此配置触发训练加速，提升训练性能。

    软件内部会根据开发者配置的加速类型、加速触发模式以及低精度训练流程占比，对相应比例的训练流程降精度编译运行，剩余的训练流程仍按照原始精度编译运行。

    该配置项取值类型为字符串，由“|”符号分割为三个字段，例如：fast|step|0.9。

  - 第一个字段代表加速类型，支持“fast”与“fast1”两种取值。
    - “fast”表示降精度时，按照float16的数据类型编译执行。
    - “fast1”表示降精度时，按照bf16的数据类型编译执行。

  - 第二个字段支持“step”和“loss”两种取值，分别表示根据step值或者loss值来分割整个训练流程为低精度训练和高精度训练。
  - 第三个字段代表低精度训练流程在总step或总loss中的占比。
    - 当第二个字段取值为“step”时，此字段的合法取值范围为“0.2 \~ 0.9”，默认值为“0.9”。
    - 当第二个字段的取值为“loss”时，此字段的合法取值范围为“1.01 \~ 1.5”，默认值为“1.05”。

    配置示例：

  - 按照step触发加速

    ```python
    accelerate_train_mode="fast|step|0.9"
    ```

  - 按照loss触发加速

    ```python
    accelerate_train_mode="fast|loss|1.05"
    ```

    **需要注意：**

    1. 若需要通过此配置项触发训练加速，需要确保网络脚本能够正常收敛。
    2. 针对网络脚本训练耗时较短的场景，若开启此配置项，端到端性能耗时不一定能够产生正向收益。
    3. 此配置项的功能与网络脚本中配置的精度模式有关：
        - 当通过“precision_mode”参数配置精度模式时，仅支持取值为"**allow_fp32_to_fp16**"， “**must_keep_origin_dtype**”或者“**空**”的场景下开启此配置。
        - 当通过“precision_mode_v2”参数配置精度模式时，仅支持取值为“**origin**”或者“**空**”的场景下开启此配置。

    4. 此配置项功能与小循环次数有关，开启小循环的场景下可能不能准确按照指定的步数或者loss值分割整个训练流程，最终可能对loss和精度产生影响。
    5. 开启此配置项时，开发者需要对网络脚本适配修改，通过[TellMeStepOrLossHook构造函数](../npu_hook/TellMeStepOrLossHook_constructor.md)通知底层软件当前执行的步数和总的步数，或者当前执行的loss和最终的目标loss。

        示例如下

        ```python
        from npu_bridge.npu_init import *
        from npu_bridge.estimator.npu.npu_config import ExperimentalConfig
        from npu_bridge.estimator.npu.npu_hook import TellMeStepOrLossHook
        # 使能fast加速模式，按照总step的0.9比例分割训练流程，即0.9占比的step数进行低精度训练，剩余step数进行高精度训练
        experimental_config = ExperimentalConfig(accelerate_train_mode="fast|step|0.9")
        config = NPURunConfig(experimental_config=experimental_config)
        est = NPUEstimator(
        model_fn=model_fn,
        config=config,
        params=params)
        hooks = []
        max_steps = 10000
        # step分割方式，告知底层当前执行的步数和总步数，值“global_step:0”仅为示例，请配置为当前步数的实际Tensor名称
        my_hook = TellMeStepOrLossHook(step='global_step:0', total_step=max_steps )
        # loss分割方式，告知底层当前执行的loss和目标loss，值“loss:0”仅为示例，请配置为当前loss的实际Tensor名称
        # my_hook = TellMeStepOrLossHook(loss='loss:0', final_loss=7.1)
        hooks.append(my_hook)
        # 开启训练
        est.train(
        input_fn=imagenet_train.input_fn,
        max_steps=max_steps,
        hooks=hooks)
        ```

## 返回值

返回ExperimentalConfig类对象，作为NPURunConfig的参数传入。

## 约束说明

无

## 调用示例

```python
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu.npu_config import ExperimentalConfig
...
experimental_config=ExperimentalConfig(accelerate_train_mode="fast|step|0.9")
session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(experimental_config=experimental_config, session_config=session_config)
```
