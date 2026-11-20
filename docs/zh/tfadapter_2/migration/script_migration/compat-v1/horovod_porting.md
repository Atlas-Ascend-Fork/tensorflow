# Horovod脚本迁移

[Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html)是基于TensorFlow、Keras、PyTorch以及MXNet的分布式训练框架，目的是提升分布式训练的性能。不同于传统的TensorFlow分布式训练采用PS worker架构，Horovod使用Allreduce来聚合梯度，能够更好地利用带宽，解决PS worker的瓶颈问题。本节介绍如何迁移基于Horovod开发的分布式训练脚本，用于在AI处理器进行分布式训练。

Horovod原始代码：

```python
import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build model...
loss = ...
opt = tf.compat.v1.train.AdagradOptimizer(0.01 * hvd.size())

# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```

迁移后的代码：

```python
import tensorflow as tf

# 导入NPU库
import npu_device as npu
from npu_device.compat.v1.npu_init import *

# 本示例调用init_resource接口，另启session进行初始化
(npu_sess, npu_shutdown) = init_resource()
npu.compat.enable_v1()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = str(get_npu_local_rank_id()) # "hvd.local_rank"修改为"get_npu_local_rank_id"

# Build model...
loss = ...
opt = tf.compat.v1.train.AdagradOptimizer(0.01 * get_npu_rank_size()) # "hvd.size"修改为"get_npu_rank_size"

# NPU allreduce
opt = npu_distributed_optimizer_wrapper(opt) # "hvd.DistributedOptimizer"修改为"npu_distributed_optimizer_wrapper"

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0')))] # "hvd.BroadcastGlobalVariablesHook(0)"修改为"NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0')))"

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if get_npu_rank_id() == 0 else None # "hvd.rank"修改为"get_npu_rank_id"

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)

# 训练结束后执行shutdown_resource, 同时关闭session
shutdown_resource(npu_sess, npu_shutdown)
close_session(npu_sess)
```
