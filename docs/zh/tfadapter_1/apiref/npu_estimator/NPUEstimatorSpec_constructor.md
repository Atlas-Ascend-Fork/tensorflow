# NPUEstimatorSpec构造函数

## 功能说明

NPUEstimatorSpec类的构造函数，NPUEstimatorSpec类继承了TensorFlow的EstimatorSpec类，可以调用基类的原生接口，定义具体的模型对象。

EstimatorSpec是model_fn的返回数据结构，包含了mode，predictions，loss，train_op和export_outputs等字段，传递到Estimator。直接使用EstimatorSpec无法满足训练的功能，定义NPUEstimatorSpec，代替EstimatorSpec的功能。

## 函数原型

```python
class NPUEstimatorSpec(model_fn_lib.EstimatorSpec):
    def __new__(cls,
                mode,
                predictions=None,
                loss=None,
                train_op=None,
                eval_metric_ops=None,
                export_outputs=None,
                training_chief_hooks=None,
                training_hooks=None,
                scaffold=None,
                evaluation_hooks=None,
                prediction_hooks=None,
                host_call=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mode | 输入 | 模式，指明当前是在训练、验证、还是推理，为继承EstimatorSpec的参数。<br><br>  - ModeKeys.TRAIN：表示训练。<br>  - ModeKeys.EVAL：表示验证。<br>  - ModeKeys.PREDICT：表示推理。|
| predictions | 输入 | 推理的输出Tensor，为继承EstimatorSpec的参数，当mode为ModeKeys.PREDICT时必须指定该参数。|
| loss | 输入 | 训练的损失，为继承EstimatorSpec的参数。 |
| train_op | 输入 | 训练算子，为继承EstimatorSpec的参数。 |
| eval_metric_ops | 输入 | 度量结果的字典（按照Tensor名称），为继承EstimatorSpec的参数。<br>字典值可以是以下之一：<br><br>  - Metric类的实例。<br>  - 调用度量函数的结果，即（metric_tensor，update_op）元组。|
| export_outputs | 输入 | 用于模型保存，描述了导出到SavedModel的输出格式，为继承EstimatorSpec的参数。 |
| training_chief_hooks | 输入 | 训练执行时主节点的SessionRunHooks集合，为继承EstimatorSpec的参数。 |
| training_hooks | 输入 | 训练执行时的SessionRunHooks集合，为继承EstimatorSpec的参数。 |
| scaffold | 输入 | 定义scaffold（提供定制saver、init_op、summary_op、global_step的能力），为继承EstimatorSpec的参数。|
| evaluation_hooks | 输入 | 验证执行时的SessionRunHooks集合，为继承EstimatorSpec的参数。 |
| prediction_hooks | 输入 | 推理执行时的SessionRunHooks集合，为继承EstimatorSpec的参数。 |
| host_call | 输入 | 捕捉Summary信息，将每个step的信息传回Host侧查看，NPUEstimatorSpec新增参数。<br>host_call是一个function和一个tensor的列表或字典组成的元组，用于返回tensor列表。<br>host_call目前适用于train()和evaluate()。|

## 返回值

返回NPUEstimatorSpec类对象。

## 调用示例

```python
from npu_bridge.npu_init import *
...
host_call = (_host_call_fn, [global_step, loss])
return NPUEstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op, host_call=host_call)
```
