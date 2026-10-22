# How Do I Reconstruct the Sess.run Distributed Script After Automated Porting?

In a distributed script of  **sess.run**, after you include the  **-d**  option to specify the distributed policy for automated porting, the porting tool cannot perform thorough porting. This is because the tool cannot identify the insertion position of  **broadcast**  and can only insert  **npu_distributed_optimizer_wrapper**  to the native gradient optimizer to implement the AllReduce function. Therefore, after the porting, you need to manually implement broadcasting.

To do so, broadcast variables using the collective communication API  broadcast after variable initialization and before training.

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
