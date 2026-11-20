# NPUCheckpointSaverHook构造函数

## 功能说明

NPUCheckpointSaverHook类的构造函数，用于保存Checkpoint文件。NPUCheckpointSaverHook类继承了CheckpointSaverHook类，可以调用基类的原生接口，用于记录训练过程中的模型信息。

## 函数原型

```python
class NPUCheckpointSaverHook(basic_session_run_hooks.CheckpointSaverHook):
    def __init__(self,
                 checkpoint_dir,
                 save_secs=None,
                 save_steps=None,
                 saver=None,
                 checkpoint_basename="model.ckpt",
                 scaffold=None,
                 listeners=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| checkpoint_dir | 输入 | 保存Checkpoint文件的路径。 |
| save_secs | 输入 | 每隔多少秒保存一次。 |
| save_steps | 输入 | 每隔多少step保存一次。 |
| saver | 输入 | Saver对象。 |
| checkpoint_basename | 输入 | Checkpoint文件的basename。 |
| scaffold | 输入 | 获取saver对象的Scaffold。 |
| listeners | 输入 | CheckpointSaverListener子类示例，用于保存checkpoint。 |

## 返回值

返回NPUCheckpointSaverHook类对象。

## 约束说明

在使用NPUEstimator并且配置iterations_per_loop\>1时，该Hook可能不生效。

## 调用示例

```python
from npu_bridge.npu_init import *
checkpoint_hook = NPUCheckpointSaverHook(checkpoint_dir='./ckpt', save_steps=2000)
...
mnist_classifier.train(
   input_fn=train_input_fn,
    steps=2000,
    hooks=[checkpoint_hook])
```
