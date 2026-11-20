# Model Construction

Build a model the same as the original model. Some code is modified for adaptation to improve compute performance. The sample code in this section shows the modifications.

## Defining Model Functions

The following uses the model function constructed based on ImageNet as an example. The related APIs are as follows.

| Class or API | Description | Location |
| --- | --- | --- |
| imagenet_model_fn() | Model function constructed based on ImageNet. | official/r1/resnet/imagenet_main.py |
| learning_rate_with_decay() | Learning rate function. When the number of global steps is less than the configured value, the learning rate increases linearly. When the number of global steps is greater than the configured value, the learning rate decreases by phase. | official/r1/resnet/resnet_run_loop.py |
| resnet_model_fn() | Constructs the EstimatorSpec class, which defines the model that is run using Estimator. | official/r1/resnet/resnet_run_loop.py |
| ImagenetModel() | Inherited from Model in the resnet_model module. It specifies the network scale, version, number of classes, convolution parameters, and pooling parameters of the ResNet model that is based on ImageNet. | official/r1/resnet/imagenet_main.py |
| __call__() | Adds more operations to classify input images, including: 1. performing NHWC to NCHW conversion to accelerate GPU computing; 2. performing the first convolution operation; 3. determining whether to perform batch normalization based on the ResNet version; 4. performing the first pooling; 5. performing block stacking; 6. computing the mean values of the input images; 7. adding fully-connected layers. | official/r1/resnet/resnet_model.py |

## Performance Improvement

1. Import the following header file to the  __official/r1/resnet/resnet_run_loop.py__  file:

    ```python
    from npu_bridge.hccl import hccl_ops
    ```

2. Check the data type of the input features or images.

   Code location:  resnet_model_fn\(\) in  official/r1/resnet/resnet_run_loop.py. The modified part is the content between  NPU modify begin and NPU modify end.

    ```python
      ############# NPU modify begin #############
        # Check whether the data type of input features or images is consistent with the data type used for computing.
      if features.dtype != dtype:
            # Change the data type of the features to dtype.
        features = tf.cast(features, dtype)
      ############## NPU modify end ###############

        # The source code is as follows.
      # assert features.dtype == dtype
    ```

3. Use the float32 type for labels to improve accuracy.

    Code location:  resnet_model_fn\(\) in official/r1/resnet/resnet_run_loop.py. The modified part is the content between  NPU modify begin  and  NPU modify end.

    ```python
        ############## NPU modify begin #############
        # Use the float32 type for labels to improve accuracy.
        accuracy = tf.compat.v1.metrics.accuracy(tf.cast(labels, tf.float32), predictions['classes'])
        ############## NPU modify end ###############

        # The accuracy computation code is as follows.
        # accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])

        accuracy_top_5 = tf.compat.v1.metrics.mean(
            tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))

        ############## NPU modify begin #############
    # Calculate accuracy during distributed training.
        rank_size = int(os.getenv('RANK_SIZE'))
        newaccuracy = (hccl_ops.allreduce(accuracy[0], "sum") / rank_size, accuracy[1])
        newaccuracy_top_5 = (hccl_ops.allreduce(accuracy_top_5[0], "sum") / rank_size, accuracy_top_5[1])
        metrics = {'accuracy': newaccuracy,
                   'accuracy_top_5': newaccuracy_top_5}
        ############## NPU modify end #############

        # Metrics in the source code is as follows.
        # metrics = {'accuracy': accuracy,
        #            'accuracy_top_5': accuracy_top_5}
    ```

4. Replace the max_pooling2d operator with max_pool_with_argmax for better compute performance.

    Code location:  call\(\)  in  official/r1/resnet/resnet_model.py. The modified part is the content between  NPU modify begin  and NPU modify end.

    ```python
         # Determine whether to perform the first pooling.
         if self.first_pool_size:
            ############## NPU modify begin #############
            # Replace max_pooling2d with max_pool_with_argmax for better performance.
            inputs,argmax = tf.compat.v1.nn.max_pool_with_argmax(
                input=inputs, ksize=(1,self.first_pool_size,self.first_pool_size,1),
                strides=(1,self.first_pool_stride,self.first_pool_stride,1), padding='SAME',
                data_format='NCHW' if self.data_format == 'channels_first' else 'NHWC')
            ############## NPU modify end ###############

                # The code uses the max_pooling2d() API for pooling.
            # inputs = tf.compat.v1.layers.max_pooling2d(
            #     inputs=inputs, pool_size=self.first_pool_size,
            #     strides=self.first_pool_stride, padding='SAME',
            #     data_format=self.data_format)

            inputs = tf.identity(inputs, 'initial_max_pool')
    ```

## Configuring Distributed Training

1. Import the following header file to the official/r1/resnet/resnet_run_loop.py  file:

    ```python
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    ```

2. Add the distributed training optimizer  __NPUDistributedOptimizer__.

    Code location:  resnet_model_fn\(\)  in  official/r1/resnet/resnet_run_loop.py. The modified part is the content between  NPU modify begin and NPU modify end.

    ```python
        if flags.FLAGS.enable_lars:
          optimizer = tf.contrib.opt.LARSOptimizer(
              learning_rate,
              momentum=momentum,
              weight_decay=weight_decay,
              skip_list=['batch_normalization', 'bias'])
        else:
          optimizer = tf.compat.v1.train.MomentumOptimizer(
              learning_rate=learning_rate,
              momentum=momentum
          )

        ############## NPU modify begin #############
        # Use the distributed training optimizer to encapsulate the single-server optimizer to support distributed training.
        # Add the following content to the code.
        optimizer = NPUDistributedOptimizer(optimizer)
        ############## NPU modify end ###############

        fp16_implementation = getattr(flags.FLAGS, 'fp16_implementation', None)
        if fp16_implementation == 'graph_rewrite':
            optimizer = (
                tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
                    optimizer, loss_scale=loss_scale))
    ```
