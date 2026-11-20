# 模型构建

模型构建与原始模型一致，部分位置经过适应性改造以提升计算性能，展示的示例代码包含改动位置。

## 定义模型函数

以根据ImageNet构建的模型函数为例，其相关函数接口如下：

| 类和接口 | 简介 | 位置 |
|----------|------|------|
| imagenet_model_fn() | 基于ImageNet构建的模型函数。 | `official/r1/resnet/imagenet_main.py` |
| learning_rate_with_decay() | 建立学习率函数，当全局步数小于设定步数时，学习率线性增加，当超过设定步数时，学习率分阶段下降。 | `official/r1/resnet/resnet_run_loop.py` |
| resnet_model_fn() | 用于构建EstimatorSpec，该类定义了由Estimator运行的模型。 | `official/r1/resnet/resnet_run_loop.py` |
| ImagenetModel() | ImagenetModel继承自resnet_model模块下的Model，指定了适用于ImageNet的ResNet模型的网络规模、版本、分类数、卷积参数和池化参数等。 | `official/r1/resnet/imagenet_main.py` |
| __call__() | 添加操作以对输入图片进行分类，包括：1、为了加速GPU运算，将输入由NHWC转换成NCHW；2、首次卷积运算；3、根据ResNet版本判断是否要做batch norm；4、首次pooling；5、堆叠block；6、计算输入图像的平均值；7、全连接层。 | `official/r1/resnet/resnet_model.py` |

## 性能提升

1. 在“official/r1/resnet/resnet_run_loop.py“增加以下头文件：

    ```python
    from npu_bridge.hccl import hccl_ops
    ```

2. 检查输入特征/图像数据类型。

    代码位置：“official/r1/resnet/resnet_run_loop.py“的resnet_model_fn\(\)函数（修改部分为“NPU modify begin”与“NPU modify end ”之间的内容）：

    ```python
      ############# NPU modify begin #############
      # 检查输入特征/图像是否与用于计算的数据类型一致。
      if features.dtype != dtype:
        # 将特征的数据类型改成与dtype一致。
        features = tf.cast(features, dtype)
      ############## NPU modify end ###############

      # 原代码中数据类型修改如下：
      # assert features.dtype == dtype
    ```

3. 计算accuracy时labels使用float32类型以提升精度。

    代码位置：“official/r1/resnet/resnet_run_loop.py“的resnet_model_fn\(\)函数（修改部分为“NPU modify begin”与“NPU modify end ”之间的内容）：

    ```python
        ############## NPU modify begin #############
        # labels使用float32类型来提升精度。
        accuracy = tf.compat.v1.metrics.accuracy(tf.cast(labels, tf.float32), predictions['classes'])
        ############## NPU modify end ###############

        # 原代码中计算accuracy如下：
        # accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])

        accuracy_top_5 = tf.compat.v1.metrics.mean(
            tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))

        ############## NPU modify begin #############
        # 用于分布式训练时的accuracy计算。
        rank_size = int(os.getenv('RANK_SIZE'))
        newaccuracy = (hccl_ops.allreduce(accuracy[0], "sum") / rank_size, accuracy[1])
        newaccuracy_top_5 = (hccl_ops.allreduce(accuracy_top_5[0], "sum") / rank_size, accuracy_top_5[1])
        metrics = {'accuracy': newaccuracy,
                   'accuracy_top_5': newaccuracy_top_5}
        ############## NPU modify end #############

        # 原代码中的metrics表示如下：
        # metrics = {'accuracy': accuracy,
        #            'accuracy_top_5': accuracy_top_5}
    ```

4. 使用max_pool_with_argmax算子替代max_pooling2d算子，以获得更好的计算性能。

    代码位置：“official/r1/resnet/resnet_model.py“的call\(\)函数（修改部分为“NPU modify begin”与“NPU modify end ”之间的内容）：

    ```python
         # 是否进行第一次池化。
         if self.first_pool_size:
            ############## NPU modify begin #############
            # 使用max_pool_with_argmax代替max_pooling2d能获得更好的表现。
            inputs,argmax = tf.compat.v1.nn.max_pool_with_argmax(
                input=inputs, ksize=(1,self.first_pool_size,self.first_pool_size,1),
                strides=(1,self.first_pool_stride,self.first_pool_stride,1), padding='SAME',
                data_format='NCHW' if self.data_format == 'channels_first' else 'NHWC')
            ############## NPU modify end ###############

            # 原代码使用max_pooling2d()接口进行池化
            # inputs = tf.compat.v1.layers.max_pooling2d(
            #     inputs=inputs, pool_size=self.first_pool_size,
            #     strides=self.first_pool_stride, padding='SAME',
            #     data_format=self.data_format)

            inputs = tf.identity(inputs, 'initial_max_pool')
    ```

## 分布式训练配置

1. 在“official/r1/resnet/resnet_run_loop.py“文件中增加以下头文件：

    ```python
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    ```

2. 添加分布式训练优化器NPUDistributedOptimizer，用于分布式训练。

    代码位置：“official/r1/resnet/resnet_run_loop.py“的resnet_model_fn\(\)函数（修改部分为“NPU modify begin”与“NPU modify end ”之间的内容）：

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
        # 使用分布式训练优化器封装单机优化器，用于支持分布式训练。
        # 在原代码中添加如下代码。
        optimizer = NPUDistributedOptimizer(optimizer)
        ############## NPU modify end ###############

        fp16_implementation = getattr(flags.FLAGS, 'fp16_implementation', None)
        if fp16_implementation == 'graph_rewrite':
            optimizer = (
                tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
                    optimizer, loss_scale=loss_scale))
    ```
