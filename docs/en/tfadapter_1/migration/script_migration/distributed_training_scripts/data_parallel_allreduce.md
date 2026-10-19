# Training in Data Parallel Mode (AllReduce)

AllReduce is a mainstream data parallel architecture. Nodes work together based on algorithms. AllReduce applies to scenarios that require high training computing power and a large number of devices. This section describes how to use the TensorFlow-based training script to perform distributed training on the  AI processor  through the AllReduce architecture.

## APIs Involved

In TensorFlow,  **tf.distribute.Strategy**  is generally used for distributed training. For details, visit  [https://www.tensorflow.org/guide/distributed_training](https://www.tensorflow.org/guide/distributed_training). Currently, the preceding distribution policy is not supported by the  AI processor. TF Adapter provides the distribution API  **npu_distributed_optimizer_wrapper**  to add the NPU AllReduce operation to the input gradient function of the optimizer and return the optimizer. In this way, gradient aggregation is performed after gradients are calculated between devices if the single-server multi-device or multi-server multi-device mode is used. After the function is called, AllReduce operators are inserted between the computed gradients and update operators in the generated training graph.

![](../../figures/data_paralle_allredce.png)

Therefore, the original TensorFlow training script needs to be updated to support distributed training on the  AI processor.

## Dataset Segmentation

During distributed training, you can use the TensorFlow APIs to split datasets. If processor resource information is required during dataset segmentation, you can obtain  AI processor  number using the collective communication API  **get_rank_size**  and obtain the processor ID using  **get_rank_id**. The following provides an example:

```bash
  dataset = dataset.shard(get_rank_size(),get_rank_id())
```

## Script Porting in Estimator Mode

1. With TensorFlow, you can pass the strategy object to  **Estimator**'s  **RunConfig**, which is not allowed by TF Adapter currently. You need to delete the related code. See the following example.

    Before porting

    ```python
    mirrored_strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(
      train_distribute=mirrored_strategy, 
      eval_distribute=mirrored_strategy,
      session_config=session_config,
      save_checkpoints_secs=60*60*24)
    ```

    After porting

    ```python
    config = NPURunConfig(
      session_config=session_config,
      save_checkpoints_secs=60*60*24)
    ```

2. Call  [npu_distributed_optimizer_wrapper](../../../apiref/npu_optimizer/npu_distributed_optimizer_wrapper.md)  to add the AllReduce operation of NPU to the input gradient function of the optimizer and return the input optimizer so that distributed computing can be implemented on the  AI processor. The specific method is as follows:

    ```python
    def cnn_model_fn(features,labels,mode):    
      # Construct the network.
      xxx    
      # Calculate the loss.
      xxx    
    
      #Configure the TrainingOp(for TRAIN mode)    
      if mode == tf.estimator.ModeKeys.TRAIN:      
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Use the SGD optimizer.
        optimizer = npu_distributed_optimizer_wrapper(optimizer) # Use NPU-based distributed computing to update gradients.
        train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step()) # Minimize the loss.
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    ```

    > [!NOTE]NOTE
    > - **NPUDistributedOptimizer**  is still compatible in the current version.
    > - In  **Estimator**  mode, when  **npu_distributed_optimizer_wrapper**  is used to implement the AllReduce function,  **NPUBroadcastGlobalVariablesHook**  is automatically added to  **NPUEstimator**. Therefore, you do not need to manually implement broadcast.

    If the original script uses the TensorFlow API to compute the gradient, for example,  **grads = tf.gradients\(loss, tvars\)**, the  **npu_allreduce**  API needs to be called to perform AllReduce on the gradient after the gradient computation is complete.

    Before porting

    ```python
    grads = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
    ```

    After porting

    ```python
    grads = npu_allreduce(tf.gradients(a + b, [a, b], stop_gradients=[a, b]))
    ```

## Script Porting in sess.run Mode

In  **Estimator**  mode, when  **npu_distributed_optimizer_wrapper**  is used to implement the AllReduce function,  **NPUBroadcastGlobalVariablesHook**  is automatically added to  **NPUEstimator**. Therefore, you do not need to manually implement broadcast. But in  **sess.run**  mode, you need manual implementation. The implementation is as follows:

1. After variable initialization and before training, broadcast variables using the  **broadcast**  collective communication API. 

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
      for var in tf.trainable_variables():
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

    In addition, the  **broadcast**  API involves graph modification. If a graph cannot be modified \(for example, the graph is frozen or a session is created using  **tf.train.Supervisor**\), you must unfreeze the graph first.

    ```python
    with sv.managed_session() as sess:
      sess.graph._unsafe_unfinalize() #  Unfreeze a graph.
      sess.run(bcast_op)
    ```

2. During training, call  **npu_distributed_optimizer_wrapper**  to aggregate the gradients after computing data of each device by using the gradient optimizer.

    ```python
    from npu_bridge.npu_init import *
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Use the SGD optimizer.
    distributedOptimizer=npu_distributed_optimizer_wrapper(optimizer) # Use NPU-based distributed computing to update gradients.
    ```

    > [!NOTE]NOTE
    > **NPUDistributedOptimizer**  is still compatible in the current version.

    If the original script uses the TensorFlow API to compute the gradient, for example,  **grads = tf.gradients\(loss, tvars\)**, the  **npu_allreduce**  API needs to be called to perform AllReduce on the gradient after the gradient computation is complete.

    Before porting

    ```python
    grads = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
    ```

    After porting

    ```python
    grads = npu_allreduce(tf.gradients(a + b, [a, b], stop_gradients=[a, b]))
    ```

## Script Porting in Keras Mode

To perform distributed training in  **Keras**  mode, modify the optimizer during  **Keras**  model compilation, call  **npu_distributed_optimizer_wrapper**  to add NPU-based AllReduce to the input gradient function of the optimizer, and add  **NPUBroadcastGlobalVariablesCallback**  to the  **callbacks**  parameter in  **keras_model.fit**.

Before porting

```python
from npu_bridge.npu_init import *

data = xxx
labels = xxx

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
keras_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy')
keras_model.fit(data, labels, epochs=10, batch_size=32) 
```

After porting

```python
from npu_bridge.npu_init import *

data = xxx
labels = xxx

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
opt = npu_distributed_optimizer_wrapper(opt)           # allreduce
keras_model.compile(optimizer=opt,loss='sparse_categorical_crossentropy')
callbacks = [NPUBroadcastGlobalVariablesCallback(0)]  # Broadcast variables.
keras_model.fit(data, labels, epochs=10, batch_size=32, callbacks=callbacks) 
```

> [!NOTE]NOTE
>**KerasDistributeOptimizer**  is still compatible in the current version.
