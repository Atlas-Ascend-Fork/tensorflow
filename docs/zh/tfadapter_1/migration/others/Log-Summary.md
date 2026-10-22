# Log/Summary

## 背景

Log和Summary算子会下沉到Device侧执行，如果用户需要捕捉Device侧的Log/Summary信息，将对应step的信息回传到Host侧查看，请参考本节内容修改训练脚本。

## 打印Log信息

Estimator模式下，系统会在Log信息回传到Host时启动dequeue线程，可直接打印出Device侧的Log信息，因此用户无需修改训练脚本：

```python
print_op = tf.print(loss)
with tf.control_dependencies([print_op]):
    train_op = xxx   # Print算子必须依赖图上能够执行到的节点，否则print节点不生效
```

> [!NOTE]说明
> NpuEstimator不支持Print算子指定output_stream，默认落盘在stderr。

而在sess.run模式下，Log信息回传到Host时不会启动dequeue线程，因此需要用户添加以下代码，单独启动dequeue线程，用于取出缓存的Log信息：

```python
from threading import Thread

import sys
def dequeue():
    tf.reset_default_graph()
    outfeed_log_tensors = npu_ops.outfeed_dequeue_op(
            channel_name="_npu_log",
            output_types=[tf.string],
            output_shapes=[()])
    dequeue_ops = tf.print(outfeed_log_tensors, output_stream=sys.stderr)
    with tf.Session() as sess:      # 可以复用训练session，也可另起session
      i = 0
      while i < max_train_steps:    # max_train_steps为最大迭代次数
        sess.run(dequeue_ops)
        i = i + 1

t1 = Thread(target=dequeue)
t1.start()
```

执行训练时，通过Assert或Print算子打印Log信息：

```python
print_op = tf.print(loss)
with tf.control_dependencies([print_op]):
    train_op = xxx   # Print算子必须依赖图上能够执行到的节点，否则Print节点不生效
```

## 打印Summary信息

sess.run模式下，暂不支持将Summary信息回传到Host侧查看。

Estimator模式下，需要用户先定义一个host_call函数，该函数中包含了用户需要采集的Summary信息。

```python
def _host_call_fn(gs, loss):
    with summary.create_file_writer(
            "./model", max_queue=1000).as_default():
        # 每个step保存一次
        with summary.always_record_summaries():
        # 每2000个step保存一次
        #with summary.record_summaries_every_n_global_steps(2000,global_step=gs):
            summary.scalar("host_call_loss", loss, step=gs)
            return summary.all_summary_ops()
```

然后通过NPUEstimatorSpec构造函数传入host_call，此时系统会在Summary算子下沉到Device侧执行时启动enqueue线程，并在Summary信息回传到Host时启动dequeue线程，用来捕捉Device侧的Summary信息，将每个或每N个step的信息传回Host侧查看。

host_call是一个function和一个tensor的列表或字典组成的元组，用于返回tensor列表，目前适用于train\(\)和evaluate\(\)。

```python
from npu_bridge.npu_init import *

host_call = (_host_call_fn, [global_step, loss])
return NPUEstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op, host_call=host_call)
```

完整代码示例：

```python
from npu_bridge.npu_init import *

# 定义一个host_call函数
from tensorflow.contrib import summary
def _host_call_fn(gs, loss):
    with summary.create_file_writer(
            "./model", max_queue=1000).as_default():
        with summary.always_record_summaries():
            summary.scalar("host_call_loss", loss, step=gs)
            return summary.all_summary_ops()

def input_fn():
     “构建dataset”

# 在model_fn中调用host_call捕捉想查看的信息
def model_fn():
     “搭建前后向模型”
  model = ***
  loss = ***
  optimizer = tf.train.MomentumOptimizer(learning_rate=c, momentum=0.9)
  global_step = tf.train.get_or_create_global_step()
  grad_vars = optimizer.compute_gradients(loss)
  minimize_op = optimizer.apply_gradients(grad_vars, global_step)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  train_op = tf.group(minimize_op, update_ops)
  host_call = (_host_call_fn, [global_step, loss])
  return NPUEstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op, host_call=host_call)

run_config = NPURunConfig()

classifier = NPUEstimator(model_fn=model_fn, config=run_config, params={ })
classifier.train(input_fn=lambda: input_fn(), max_steps=1000)
```
