# NPUOutputTensorHook构造函数

## 功能说明

NPUOutputTensorHook类的构造函数，NPUOutputTensorHook作用于NPUEstimator的train、evaluate、predict流程中的Hook，用于每N步或者结束时调用用户自定义的output_fn，打印输出tensors。NPUOutputTensorHook类继承了LoggingTensorHook类，可以调用基类的原生接口。

## 函数原型

```python
class NPUOutputTensorHook(basic_session_run_hooks.LoggingTensorHook):
    def __init__(self, tensors,
                 dependencies=None,
                 output_fn=None,
                 output_every_n_steps=0
                 )
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensors | 输入 | 输入tensor的名称集合。字典或列表格式。 |
| dependencies | 输入 | tensors对应的依赖。 |
| output_fn | 输入 | tensors的输出打印函数。 |
| output_every_n_steps | 输入 | 会话执行N次和训练脚本执行结束时调用用户定义的output_fn。 |

## 返回值

返回NPUOutputTensorHook类对象。

## 约束说明

iterations_per_loop\>1的场景下，无法按照output_every_n_steps指定的值调用output_fn。

## 调用示例

```python
from npu_bridge.npu_init import *

# 定义output_fn
def output_fn(inputs):
  device_id = os.environ["ASCEND_DEVICE_ID"]
  output_file = os.path.join("/code", device_id, "test_npu_output_tensor.txt")
  for item in inputs:
    content = "step:{},loss:{}".format(str(item['global_step']), str(item['loss']))
    with open(output_file, 'a') as f:
      f.write(content)
      f.write("\n")

# 定义output_hook，用于调用用户定义的output_fn
  tensors = {'global_step': global_step, 'loss': loss}
  output_hook = NPUOutputTensorHook(
      tensors,
      dependencies=train_op_list,
      output_fn=output_fn,
      output_every_n_steps=10)
  train_hook.append(output_hook)

# 在EstimatorSpec传入hook
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      training_chief_hooks=train_hook,
      eval_metric_ops=metrics)
```
