# How Do I Modify the Estimator Single-Device Training Script to a Distributed Training Script?

The porting tool is able to directly port Estimator distributed training scripts, which can be used for distributed training after porting. For a single-device training script, it cannot be directly used for distributed training after tool-based porting. Manual tweaks are needed.

Single-device training script after tool-based porting:

```python
def cnn_model_fn(features,labels,mode):
  # Construct the network.
  xxx
  # Calculate the loss.
  xxx
  #Configure the TrainingOp(for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Use the SGD optimizer.
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step()) # Minimize the loss.
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
...
hook=hk._LoggerHook(FLAGS)
training_hooks = []
training_hooks.append(hook)
...
estimator.train(train_data_fn, max_steps=num_steps // rank_size, hooks=training_hooks)
```

Modified script for distributed training \(using method 1\):

```python
def cnn_model_fn(features,labels,mode):
  # Construct the network.
  xxx
  # Calculate the loss.
  xxx
  #Configure the TrainingOp(for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = npu_distributed_optimizer_wrapper(optimizer) # Update the gradient.
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
...
hook=hk._LoggerHook(FLAGS)
training_hooks = []
training_hooks.append(hook)
training_hooks.append(NPUBroadcastGlobalVariablesHook(0,int(os.getenv('RANK_ID','0')))) # Broadcast variables.
...
estimator.train(train_data_fn, max_steps=num_steps, hooks=training_hooks)
```

Modified script for distributed training \(using method 2\):

```python
def cnn_model_fn(features,labels,mode):
  # Construct the network.
  xxx
  # Calculate the loss.
  xxx
  #Configure the TrainingOp(for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = npu_distributed_optimizer_wrapper(optimizer) # Update the gradient.
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
...
hook=hk._LoggerHook(FLAGS)
training_hooks = []
training_hooks.append(hook)
...
estimator.train(train_data_fn, max_steps=num_steps, hooks=npu_hooks_append(training_hooks))  # Broadcast variables.
```
