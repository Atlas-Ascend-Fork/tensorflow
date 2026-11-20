# Porting with Horovod

[Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html)  is a distributed training framework for TensorFlow, Keras, PyTorch, and MXNet, aiming to improve distributed training performance. Compared with the traditional TensorFlow distributed training that uses the PS-Worker architecture, Horovod uses AllReduce to aggregate gradients to better use the bandwidth and eliminate the PS-Worker bottleneck. This section describes how to port your distributed training script developed based on Horovod for distributed training on the  AI processor.

Original Horovod code:

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

Code after porting:

```python
import tensorflow as tf

# Import the NPU library.
import npu_device as npu
from npu_device.compat.v1.npu_init import *

# In this example, the init_resource API is called to start another session for initialization.
(npu_sess, npu_shutdown) = init_resource()
npu.compat.enable_v1()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = str(get_npu_local_rank_id()) # Change "hvd.local_rank" to "get_npu_local_rank_id".

# Build model...
loss = ...
opt = tf.compat.v1.train.AdagradOptimizer(0.01 * get_npu_rank_size()) # Change "hvd.size" to "get_npu_rank_size".

# NPU allreduce
opt = npu_distributed_optimizer_wrapper(opt) # Change "hvd.DistributedOptimizer" to "npu_distributed_optimizer_wrapper".

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0')))] # Change "hvd.BroadcastGlobalVariablesHook(0)" to "NPUBroadcastGlobalVariablesHook(0, int(os.getenv('RANK_ID', '0')))".

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if get_npu_rank_id() == 0 else None # Change "hvd.rank" to "get_npu_rank_id".

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)

# After the training is complete, run shutdown_resource and close the session.
shutdown_resource(npu_sess, npu_shutdown)
close_session(npu_sess)
```
