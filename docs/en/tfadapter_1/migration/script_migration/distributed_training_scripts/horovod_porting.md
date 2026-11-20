# Porting with Horovod

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and MXNet, aiming to improve distributed training performance. Compared with the traditional TensorFlow distributed training that uses the PS-Worker architecture, Horovod uses AllReduce to aggregate gradients to utilize the bandwidth and solve the bottleneck of PS-Worker. This section describes how to port your distributed training script developed based on Horovod for distributed training on NPUs.

For details about Horovod, visit  [Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html).

## Mapping Between Horovod APIs and NPU APIs

The following table lists the mapping between common Horovod TensorFlow APIs and NPU APIs.

| Horovod API | Description | NPU API |
| --- | --- | --- |
| horovod.tensorflow.init | Horovod initialization | npu_ops.initialize_system |
| horovod.tensorflow.shutdown | Horovod shutdown | npu_ops.shutdown_system |
| horovod.tensorflow.DistributedOptimizer | Distributed optimizer | npu_distributed_optimizer_wrapper |
| horovod.tensorflow.size | Number of global ranks | get_rank_size |
| horovod.tensorflow.local_size | Number of ranks of the current server | get_local_rank_size |
| horovod.tensorflow.rank | Global rank ID | get_rank_id |
| horovod.tensorflow.local_rank | Rank ID of the current server | get_local_rank_id |
| horovod.tensorflow.allgather | AllGather operation | hccl_ops.allgather |
| horovod.tensorflow.broadcast | Broadcast operation | hccl_ops.broadcast |
| horovod.tensorflow.alltoall | AlltoAll operation | hccl_ops.all_to_all_v |
| horovod.tensorflow.allreduce | AllReduce operation | hccl_ops.allreduce |

## Porting Example

Original Horovod code

```python
import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

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
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```

Code after porting:

```python
# Import the NPU libraries.
import tensorflow as tf
from npu_bridge.npu_init import *

# In this example, another session is created to initialize HCCL when the HCCL group management API is called.
npu_int = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()
config = tf.ConfigProto(allow_soft_placement=True)
# Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
# Explicitly disable the remapping and memory_optimization functions of TensorFlow to avoid conflicts with the functions of the NPU.
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
init_sess = tf.Session(config=config)
init_sess.run(npu_int)

# Pin GPU to be used to process local rank (one GPU per process)
config.gpu_options.visible_device_list = str(get_local_rank_id())  # Change "hvd.local_rank" to "get_local_rank_id".

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * get_rank_size())   # Change "hvd.size" to "get_rank_size".

# NPU allreduce
# Change hvd.DistributedOptimizer to npu_distributed_optimizer_wrapper.
opt = npu_distributed_optimizer_wrapper(opt)
# Add hook to broadcast variables from rank 0 to all other processes during initialization.
hooks = [NPUBroadcastGlobalVariablesHook(0)]

# In sess.run mode, call the broadcast collective communication API to broadcast variables.
input = tf.trainable_variables()
bcast_global_variables_op = hccl_ops.broadcast(input, 0)

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if get_rank_id() == 0 else None  # Change "hvd.rank" to "get_rank_id".

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  # Broadcast the variables.
  mon_sess.run(bcast_global_variables_op)
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)

# After training, call the shutdown_system function to close the session.
init_sess.run(npu_shutdown)
init_sess.close()
```

> [!NOTE]NOTE
> **NPUDistributedOptimizer**  is still compatible in the current version.
