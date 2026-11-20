# set_iteration_per_loop

## Description

Sets the number of iterations per training loop in  **sess.run**  mode, that is, the number of training iterations executed on the device side in each  **sess.run\(\)**  call. This API can save unnecessary interactions between the host and device and reduce the training time consumption.

## Prototype

```python
def set_iteration_per_loop(sess, train_op, iterations_per_loop=1)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sess | Input | Created TensorFlow session |
| train_op | Input | Operation that updates the gradient |
| iterations_per_loop | Input | Number of iterations per training loop per sess.run() call on the device side. Defaults to 1. The total number of iterations per training loop must be an integer multiple of iterations_per_loop.<br>In mixed compute mode (mix_compile_mode is set to True), iterations_per_loop must be set to 1. |

## Returns

An operator for the user to call by using  **sess.run\(op\)**

## Restrictions

The preceding API involves graph modification. If a graph cannot be modified \(for example, the graph is frozen or a session is created using  **tf.train.Supervisor**\), you cannot use the  **set_iteration_per_loop**  API to set the loops and iterations per loop. In this case, use  [create_iteration_per_loop_var](create_iteration_per_loop_var.md)  and  [load_iteration_per_loop_var](load_iteration_per_loop_var.md).

## Example

```python
from npu_bridge.npu_init import *

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["enable_data_pre_proc"].b = True # If the GetNext operator exists on the network, offload it. GetNext operator offload is a prerequisite for iteration offload.
custom_op.parameter_map["iterations_per_loop"].i = 10 # Determine whether the training iteration is offloaded. Must be equal to iterations_per_loop set in set_iteration_per_loop.
config = npu_config_proto(config_proto=config)

# Train the model.
with tf.Session(config=config) as sess:
    sess.run(init)
    # Set the number of iterations per loop to 10 in sess.run mode.
    train_op = util.set_iteration_per_loop(sess, optimizer, 10)  # sess indicates the TensorFlow session, optimizer indicates the gradient update operation, and 10 indicates the number of training iterations on the device.

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```
