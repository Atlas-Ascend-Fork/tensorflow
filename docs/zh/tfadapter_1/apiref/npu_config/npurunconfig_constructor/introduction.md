# 简介

## 功能说明

开发者在AI处理器上通过Estimator模式进行模型训练或在线推理时，可通过NPURunConfig类的构造函数，指定Estimator的运行配置。

NPURunConfig类继承了tf.estimator的RunConfig类，关于对RunConfig类原生接口的支持情况可参见[RunConfig参数支持说明](../runconfig_params_support_info.md)。

## 函数原型

您可以在TensorFlow Adapter软件安装路径下的：python/site-packages/npu_bridge/estimator/npu/npu_config.py文件中查看NPURunConfig的原型定义，示例如下：

```python
class NPURunConfig(run_config_lib.RunConfig):
    def __init__(self,
                 iterations_per_loop=1,
                 profiling_config=None,
                 model_dir=None,
                 tf_random_seed=None,
                 save_summary_steps=0,
                 save_checkpoints_steps=None,
                 save_checkpoints_secs=None,
                 ...
                 )
```

NPURunConfig支持的详细参数请以后面章节的参数说明为准。

## 使用约束

使用多Device执行训练的场景下，不支持使用按时间保存文件的参数save_checkpoints_secs。

## 返回值

返回NPURunConfig类对象，作为NPUEstimator的初始化参数传入。

## 调用示例

NPURunConfig配置的通用使用方式如下所示：

```python
from npu_bridge.npu_init import *
session_config=tf.ConfigProto()
config = NPURunConfig(
    session_config=session_config,
    mix_compile_mode=False,
    iterations_per_loop=1000)
```
