# Log and Summary Operators

## Background

The execution of Log and Summary operators is offloaded to the device side. If you need to capture the Log and Summary information on the device side and view the information of the corresponding step on the host side, modify the training script by referring to this section.

## Log Printing

In  **Estimator**  mode, the system starts the dequeue thread when the Log information is returned to the host. The Log information on the device side can be directly printed. Therefore, no modification is needed.

```python
print_op = tf.print(loss)
with tf.control_dependencies([print_op]):
    train_op = xxx   # The Print operator depends on the nodes that can be executed on the graph. Otherwise, the Print operator does not take effect.
```

> [!NOTE]NOTE
> **NpuEstimator**  does not support the Print operator to specify  **output_stream**. By default, data is flushed to  **stderr**.

In  **sess.run**  mode, the dequeue thread is not started when the log information is returned to the host. Therefore, you can add the following code to start the dequeue thread separately to obtain the buffered Log information.

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
    with tf.Session() as sess:      # You can reuse the training session or start another session.
      i = 0
      while i < max_train_steps:    # max_train_steps indicates the maximum number of iterations.
        sess.run(dequeue_ops)
        i = i + 1

t1 = Thread(target=dequeue)
t1.start()
```

For training, the Assert or Print operator is used to print Log information.

```python
print_op = tf.print(loss)
with tf.control_dependencies([print_op]):
    train_op = xxx   # The Print operator depends on the nodes that can be executed on the graph. Otherwise, the Print operator does not take effect.
```

## Summary Printing

In  **sess.run**  mode, it is not supported to send Summary information back to the host for viewing.

In  **Estimator**  mode, you need to define a  **host_call**  function that contains the Summary information to be collected.

```python
def _host_call_fn(gs, loss):
    with summary.create_file_writer(
            "./model", max_queue=1000).as_default():
        # Record summary every step.
        with summary.always_record_summaries():
        # Record summary every 2000 steps.
        #with summary.record_summaries_every_n_global_steps(2000,global_step=gs):
            summary.scalar("host_call_loss", loss, step=gs)
            return summary.all_summary_ops()
```

Then, pass  **host_call**  to the  **NPUEstimatorSpec**  constructor. The system starts the enqueue thread when the Summary operator is executed on the device side and starts the dequeue thread when the Summary information is sent back to the host, so that the information of each or  _N_  steps will be sent back to the host.

**host_call**  is a tuple consisting of a function and a list or dictionary of tensors. It is used to return a list of tensors. **host_call** applies to **train\(\)**  and **evaluate\(\)** calls.

```python
from npu_bridge.npu_init import *

host_call = (_host_call_fn, [global_step, loss])
return NPUEstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op, host_call=host_call)
```

The following is a complete code example.

```python
from npu_bridge.npu_init import *

# Define a host_call function.
from tensorflow.contrib import summary
def _host_call_fn(gs, loss):
    with summary.create_file_writer(
            "./model", max_queue=1000).as_default():
        with summary.always_record_summaries():
            summary.scalar("host_call_loss", loss, step=gs)
            return summary.all_summary_ops()

def input_fn():
       "Build dataset"

# Call host_call in model_fn to capture the information to be viewed.
def model_fn():
       "Build a forward/backward model"
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
