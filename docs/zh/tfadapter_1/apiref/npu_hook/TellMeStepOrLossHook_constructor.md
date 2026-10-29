# TellMeStepOrLossHook构造函数

## 功能说明

TellMeStepOrLossHook类的构造函数，TellMeStepOrLossHook用于告知底层软件“当前执行的步数和总的步数”或者“当前执行的loss和最终的目标loss”。

## 函数原型

```python
class TellMeStepOrLossHook(session_run_hook.SessionRunHook):
    def __init__(self, step=None, total_step=None, loss=None, final_loss=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| step | 输入 | 表示当前步数的Tensor的名称。 |
| total_step | 输入 | 训练脚本总的训练步数。 |
| loss | 输入 | 表示当前loss的Tensor的名称。 |
| final_loss | 输入 | 训练脚本最终的目标loss。 |

## 返回值

返回TellMeStepOrLossHook类对象。

## 约束说明

Iterations_per_loop\>1的场景下，会按照每增加Iterations_per_loop数量的步数，就会告知底层软件当前执行的步数或者loss，无法做到每增加1步就告知底层软件一次，可能对底层软件某些依赖此hook函数结果的功能产生影响。

## 调用示例

```python
from npu_bridge.npu_init import *
est = NPUEstimator(
        model_fn=model_fn,
        config=config,
        params=params)
hooks = []
max_steps = 10000
# step分割的方式，本示例当前step的tensor名称是global_step:0，总step数是10000，请根据实际step的tensor名称和总step数进行配置
my_hook = TellMeStepOrLossHook(step='global_step:0', total_step=max_steps)
# loss分割的方式，本示例当前loss的tensor名称是loss:0，目标loss是7.1，请根据实际loss的tensor名称和目标loss值进行配置
# my_hook = TellMeStepOrLossHook(loss='loss:0', final_loss=7.1)
hooks.append(my_hook)
# 开启训练
est.train(
          input_fn=imagenet_train.input_fn,
          max_steps=max_steps,
          hooks=hooks)
```
