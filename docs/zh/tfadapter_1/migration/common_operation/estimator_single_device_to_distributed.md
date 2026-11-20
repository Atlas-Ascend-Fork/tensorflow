# Estimator单卡训练脚本改造为分布式训练脚本

对于Estimator的分布式脚本，使用迁移工具可支持直接迁移成分布式脚本。如果有用户原始脚本是单卡训练脚本，迁移工具迁移后的脚本并不能够进行分布式训练，但用户可以基于迁移后的脚本，通过少量手工修改使其支持分布式训练。

工具迁移后的单机脚本：

```python
def cnn_model_fn(features,labels,mode):
  #搭建网络
  xxx
  #计算loss
  xxx
  #Configure the TrainingOp(for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # 使用SGD优化器
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step()) # 最小化loss
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
...
hook=hk._LoggerHook(FLAGS)
training_hooks = []
training_hooks.append(hook)
...
estimator.train(train_data_fn, max_steps=num_steps // rank_size, hooks=training_hooks)
```

手工修改后支持分布式训练（方法一）：

```python
def cnn_model_fn(features,labels,mode):
  #搭建网络
  xxx
  #计算loss
  xxx
  #Configure the TrainingOp(for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = npu_distributed_optimizer_wrapper(optimizer) # 梯度更新
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
...
hook=hk._LoggerHook(FLAGS)
training_hooks = []
training_hooks.append(hook)
training_hooks.append(NPUBroadcastGlobalVariablesHook(0,int(os.getenv('RANK_ID','0')))) # 变量广播
...
estimator.train(train_data_fn, max_steps=num_steps, hooks=training_hooks)
```

手工修改后支持分布式训练（方法二）：

```python
def cnn_model_fn(features,labels,mode):
  #搭建网络
  xxx
  #计算loss
  xxx
  #Configure the TrainingOp(for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = npu_distributed_optimizer_wrapper(optimizer) # 梯度更新
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
...
hook=hk._LoggerHook(FLAGS)
training_hooks = []
training_hooks.append(hook)
...
estimator.train(train_data_fn, max_steps=num_steps, hooks=npu_hooks_append(training_hooks))  # 变量广播
```
