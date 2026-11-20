# 执行训练

## 训练模块

训练模块函数接口如下：

| 接口 | 简介 | 位置 |
|------|------|------|
| main() | 主函数入口，包含训练之前集合通信初始化，执行模型训练。 | `official/r1/resnet/imagenet_main.py` |
| run_imagenet() | 模型训练入口，负责输入函数选择，并返回训练结果。 | `official/r1/resnet/imagenet_main.py` |
| resnet_main() | 包含运行配置、训练以及验证的主要函数。 | `official/r1/resnet/resnet_run_loop.py` |

## 分布式训练

1. 在“official/r1/resnet/resnet_run_loop.py“增加以下头文件：

    ```python
    from npu_bridge.estimator import npu_ops
    from tensorflow.core.protobuf import rewriter_config_pb2
    ```

2. 训练之前集合通信初始化。

    代码位置：“official/r1/resnet/imagenet_main.py“的main\(\)函数（修改部分如下）：

    ```python
    def main(_):
        ############## NPU modify begin #############
        # 初始化NPU，调用HCCL接口。
        # 在原代码中添加如下内容：
        init_sess, npu_init = resnet_run_loop.init_npu()
        init_sess.run(npu_init)
        ############## NPU modify end ###############

        with logger.benchmark_context(flags.FLAGS):
            run_imagenet(flags.FLAGS)
    ```

3. 集合通信初始化函数定义。

    代码位置：“official/r1/resnet/resnet_run_loop.py“中添加init_npu\(\)函数接口：

    ```python
    def resnet_main(flags_obj, model_function, input_function, dataset_name, shape=None):...
    ############## NPU modify begin #############
    # 添加如下代码
    def init_npu():
        """此接口作用是手动初始化NPU。
        返回值：
          `init_sess` npu  init session config.
          `npu_init` npu  init ops.
        """
        npu_init = npu_ops.initialize_system()
        config = tf.ConfigProto()

        config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # custom_op.parameter_map["precision_mode"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["use_off_line"].b = True

        init_sess = tf.Session(config=config)
        return init_sess, npu_init
    ############## NPU modify end ###############
    ```

4. 单次训练/验证结束后释放设备资源。

    代码位置：“official/r1/resnet/resnet_run_loop.py“中的resnet_main\(\)函数中（修改部分为“NPU modify begin”与“NPU modify end ”之间的内容）。

    ```python
    for cycle_index, num_train_epochs in enumerate(schedule):
          tf.compat.v1.logging.info('Starting cycle: %d/%d', cycle_index,
                                    int(n_loops))

          if num_train_epochs:
            # Since we are calling classifier.train immediately in each loop, the
            # value of num_train_epochs in the lambda function will not be changed
            # before it is used. So it is safe to ignore the pylint error here
            # pylint: disable=cell-var-from-loop
            classifier.train(
                input_fn=lambda input_context=None: input_fn_train(
                    num_train_epochs, input_context=input_context),
                hooks=train_hooks,
                max_steps=flags_obj.max_train_steps)

          ############## NPU modify begin #############
          # 单次训练结束时，释放NPU资源，在下一次进程开始之前如果要用到HCCL接口需要重新初始化。
          # 在原代码中添加如下内容:
          init_sess, npu_init = init_npu()
          npu_shutdown = npu_ops.shutdown_system()
          init_sess.run(npu_shutdown)
          init_sess.run(npu_init)
          ############## NPU modify end ###############

          tf.compat.v1.logging.info('Starting to evaluate.')
          eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                             steps=flags_obj.max_train_steps)

          benchmark_logger.log_evaluation_result(eval_results)

          if model_helpers.past_stop_threshold(
              flags_obj.stop_threshold, eval_results['accuracy']):
            break

          ############## NPU modify begin #############
          # 单次训练结束时，释放NPU资源，在下一次进程开始之前如果要用到HCCL接口需要重新初始化。
          # 在原代码中添加如下内容:
          init_sess, npu_init = init_npu()
          npu_shutdown = npu_ops.shutdown_system()
          init_sess.run(npu_shutdown)
          init_sess.run(npu_init)
          ############## NPU modify end ###############
    ```

5. 所有训练/验证结束后释放设备资源。

    在所有训练/验证结束后通过npu_ops.shutdown_system接口释放设备资源。

    代码位置：“official/r1/resnet/resnet_run_loop.py“中的resnet_main\(\)函数中（修改部分如下）：

    ```python
        if flags_obj.export_dir is not None:
            # Exports a saved model for the given classifier.
            export_dtype = flags_core.get_tf_dtype(flags_obj)
            if flags_obj.image_bytes_as_serving_input:
                input_receiver_fn = functools.partial(
                    image_bytes_serving_input_fn, shape, dtype=export_dtype)
            else:
                input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
                    shape, batch_size=flags_obj.batch_size, dtype=export_dtype)
            classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn,
                                         strip_default_attrs=True)

        ############## NPU modify begin #############
        # 在所有训练/验证结束后通过npu_ops.shutdown_system接口释放设备资源。在原代码中添加如下内容：
        npu_shutdown = npu_ops.shutdown_system()
        init_sess.run(npu_shutdown)
        ############## NPU modify end ###############

        stats = {}
        stats['eval_results'] = eval_results
        stats['train_hooks'] = train_hooks

        return stats
    ```

## 补充说明（性能调优）

设定Loss Scale默认值。

代码位置：“official/r1/resnet/imagenet_main.py“的define_imagenet_flags\(\)函数（修改部分如下）：

```python
def define_imagenet_flags():
    resnet_run_loop.define_resnet_flags(
        resnet_size_choices=['18', '34', '50', '101', '152', '200'],
        dynamic_loss_scale=True,
        fp16_implementation=True)
    flags.adopt_module_key_flags(resnet_run_loop)
    flags_core.set_defaults(train_epochs=90)

    ############## NPU modify begin #############
    # 昇腾AI处理器默认支持混合精度训练，loss_scale设置过大可能导致梯度爆炸，设置过小可能会导致梯度消失，
    # 设置以下经验值能避免以上问题。
    flags_core.set_defaults(loss_scale='512')
    ############## NPU modify end ###############
```
