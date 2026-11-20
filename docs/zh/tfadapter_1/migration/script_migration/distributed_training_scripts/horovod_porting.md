# Horovod脚本迁移

Horovod是基于TensorFlow、Keras、PyTorch以及MXNet的分布式训练框架，目的是提升分布式训练的性能。不同于传统的TensorFlow分布式训练采用PS-Worker架构，Horovod使用AllReduce进行聚合梯度，能够更好地利用带宽，解决PS-Worker的瓶颈问题。本节介绍如何迁移基于Horovod开发的分布式训练脚本，使其在NPU进行分布式训练。

关于Horovod的介绍，可参见[Horovod](https://horovod.readthedocs.io/en/stable/tensorflow.html)官网。

## Horovod API与NPU API对应关系

常用的Horovod TensorFlow API与NPU API的对应关系如下表所示：

| Horovod API | 描述 | NPU API |
| --- | --- | --- |
| horovod.tensorflow.init | Horovod初始化 | npu_ops.initialize_system |
| horovod.tensorflow.shutdown | Horovod销毁 | npu_ops.shutdown_system |
| horovod.tensorflow.DistributedOptimizer | 分布式优化器 | npu_distributed_optimizer_wrapper |
| horovod.tensorflow.size | 返回全局的rank个数 | get_rank_size |
| horovod.tensorflow.local_size | 返回当前server的rank个数 | get_local_rank_size |
| horovod.tensorflow.rank | 返回全局rank id | get_rank_id |
| horovod.tensorflow.local_rank | 返回当前server的rank id | get_local_rank_id |
| horovod.tensorflow.allgather | allgather操作 | hccl_ops.allgather |
| horovod.tensorflow.broadcast | broadcast操作 | hccl_ops.broadcast |
| horovod.tensorflow.alltoall | alltoall操作 | hccl_ops.all_to_all_v |
| horovod.tensorflow.allreduce | allreduce操作 | hccl_ops.allreduce |

## 迁移示例

Horovod原始代码：

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

迁移后的代码：

```python
# 导入NPU库
import tensorflow as tf
from npu_bridge.npu_init import *

# 本示例调用了HCCL的group管理接口，因此需要另起session进行HCCL初始化。
npu_int = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()
config = tf.ConfigProto(allow_soft_placement=True)
# 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
# 需要显示关闭TensorFlow的remapping、memory_optimization功能，避免与NPU中的功能冲突。
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
init_sess = tf.Session(config=config)
init_sess.run(npu_int)

# Pin GPU to be used to process local rank (one GPU per process)
config.gpu_options.visible_device_list = str(get_local_rank_id())  # "hvd.local_rank"修改为"get_local_rank_id"

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01 * get_rank_size())   # "hvd.size"修改为"get_rank_size"

# NPU allreduce
# 将"hvd.DistributedOptimizer"修改为"npu_distributed_optimizer_wrapper"
opt = npu_distributed_optimizer_wrapper(opt)
# Add hook to broadcast variables from rank 0 to all other processes during initialization.
hooks = [NPUBroadcastGlobalVariablesHook(0)]

# 在session run模式下调用集合通信接口broadcast进行变量广播：
input = tf.trainable_variables()
bcast_global_variables_op = hccl_ops.broadcast(input, 0)

# Make training operation
train_op = opt.minimize(loss)

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if get_rank_id() == 0 else None  # "hvd.rank"修改为"get_rank_id"

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  # 变量广播
  mon_sess.run(bcast_global_variables_op)
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)

# 训练结束后执行shutdown_system，同时关闭session
init_sess.run(npu_shutdown)
init_sess.close()
```

> [!NOTE]说明
> NPUDistributedOptimizer分布式优化器在当前版本依然兼容。
