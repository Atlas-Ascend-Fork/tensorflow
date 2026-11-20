# How Do I Fix the Training Error Caused by Placing Variable Initialization and Data Preprocessing Initialization in the Same Subgraph?

## Symptom

Some networks such as LeNet ported to  AI processor  for training are found with no loss convergence and an unsatisfying accuracy.

![](../figures/lenet_loss_faq.png)

In addition, it is found that all variables are initialized to 0s.

## Possible Cause

The user training script is analyzed as follows.

```python
sess.run(tf.group(
    tf.global_variables_initializer(),  # Variable initialization
    tf.local_variables_initializer(),   # Variable initialization
    iterator.initializer                # Data preprocessing initialization
    ))
```

Based on the preceding, in TensorFlow graph partitioning, variable initialization and data preprocessing initialization use the same subgraph where only variable initialization can be offloaded to the device. As a result, the entire subgraph cannot be offloaded to the  AI processor.

Because variable initialization is performed on the host, all variables are initialized to zero.

## Solution

Schedule variable initialization and data preprocessing initialization to different subgraphs during TensorFlow graph partitioning.

```python
sess.run(tf.group(
    tf.global_variables_initializer(),  # Variable initialization
    tf.local_variables_initializer()    # Variable initialization
    ))
sess.run(iterator.initializer)        # Data preprocessing initialization
```
