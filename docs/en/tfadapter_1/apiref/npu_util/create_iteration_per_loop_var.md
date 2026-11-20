# create_iteration_per_loop_var

## Description

This API is used in conjunction with  [load_iteration_per_loop_var](load_iteration_per_loop_var.md)  to set the number of iterations per training loop every  **sess.run\(\)**  call on the device side. This API is used to modify a graph and set the number of iterations per loop using  [load_iteration_per_loop_var](load_iteration_per_loop_var.md).

## Prototype

```python
def create_iteration_per_loop_var(self, train_op)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| train_op | Input | Operation that updates the gradient |

## Returns

An operator for the user to call by using  **sess.run\(op\)**

## Example

```python
from npu_bridge.npu_init import *

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["enable_data_pre_proc"].b = True # If the GetNext operator exists on the network, offload it. GetNext operator offload is a prerequisite for iteration offload.
custom_op.parameter_map["iterations_per_loop"].i = 10  # Used for functional validation. Must be equal to iterations_per_loop set in load_iteration_per_loop_var.
config = npu_config_proto(config_proto=config)

# Train a model.
with tf.Session(config=config) as sess:
    sess.run(init)
    # Set the number of iterations per loop to 10 in sess.run mode.
    iteration = util.IterationPerLoop()   # Define the IterationPerLoop object.
    train_op = iteration.create_iteration_per_loop_var(optimizer) # Modify the graph.
    tf.train.Supervisor(logdir="/home/xxxx",init_op=init)  # Freeze the graph.
    iteration.load_iteration_per_loop_var(sess, 10)  # Set the number of iterations per loop.

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```
