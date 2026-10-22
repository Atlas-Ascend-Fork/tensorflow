# 自动迁移后如何进行sess.run分布式脚本改造

对于sess.run的分布式脚本，用户输入-d参数指定分布式策略完成自动迁移后，迁移工具不能进行彻底迁移，原因是：工具无法识别broadcast的插入位置，仅能对原生梯度优化器插入npu_distributed_optimizer_wrapper实现allreduce的功能，因此工具迁移后，需要用户手写实现broadcast功能。

具体方法为，在变量初始化之后，训练之前，通过集合通信接口broadcast进行变量广播。

```python
from npu_bridge.npu_init import *

def broadcast_global_variables(root_rank, index):
  """Broadcasts all global variables from root rank to all other processes.
  Arguments:
  root_rank: rank of the process from which global variables will be broadcasted
  to all other processes.
  index: rank_id
  """
  op_list = []
  for var in tf.global_variables():
    # the input and out tensor of HCOMBroadcast interface are list
    if "float" in var.dtype.name:
      inputs = [var]
      outputs=hccl_ops.broadcast(tensor=inputs,root_rank=root_rank)
      if outputs is not None:
        op_list.append(outputs[0].op)
        op_list.append(tf.assign(var, outputs[0]))
  return tf.group(op_list)

...
bcast_op = broadcast_global_variables(root_rank, index)
sess = tf.Session()
...
sess.run(bcast_op)
```
