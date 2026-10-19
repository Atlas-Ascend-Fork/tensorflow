# NPUEstimator构造函数

## 功能说明

NPUEstimator类的构造函数，NPUEstimator类继承了TensorFlow的Estimator类，可以调用基类的原生接口，用来训练、评估、推理TensorFlow模型。

## 函数原型

```python
class NPUEstimator(estimator_lib.Estimator):
    def __init__(self,
                 model_fn=None,
                 model_dir=None,
                 config=None,
                 params=None,
                 job_start_file='',
                 warm_start_from=None
                 )
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| model_fn | 输入 | 模型function定义，该function返回NPUEstimatorSpec类对象。<br>关于NPUEstimatorSpec类的构造函数，请参见[NPUEstimatorSpec构造函数](NPUEstimatorSpec_constructor.md)。 |
| model_dir | 输入 | 保存模型路径，用于保存或恢复模型文件。默认为None。<br>如果NPURunConfig和NPUEstimator配置的model_dir不同，系统报错。<br>如果NPURunConfig和NPUEstimator仅一个接口配置model_dir，以配置的路径为准。<br>如果NPURunConfig和NPUEstimator均未配置model_dir，则系统在当前脚本执行路径创建一个model_dir_xxxxxxxxxx目录保存模型文件。 |
| config | 输入 | NPURunConfig类对象。<br>关于NPURunConfig类的构造函数，请参见[NPURunConfig构造函数](../npu_config/npurunconfig_constructor/README.md)。 |
| params | 输入 | 传入model_fn的参数，为字典类型，键为传入参数的名字，值为基本的Python类型值 |
| job_start_file | 输入 | CSA job启动文件路径。 |
| warm_start_from | 输入 | 指定checkpoint路径，会导入该checkpoint开始训练。 |

## 返回值

返回NPUEstimator类对象。

## 调用示例

```python
from npu_bridge.npu_init import *
...
self._classifier=NPUEstimator(
  model_fn=cnn_model_fn,
  model_dir=self._model_dir,
  config=NPURunConfig(
      save_checkpoints_steps=50 if get_rank_id() == 0 else 0,
      keep_checkpoint_max=1))
```
