# Performing Training

## Training Module

The following table describes the APIs of the training module.

| Function | Description | Location |
| --- | --- | --- |
| main() | Main function entry, for collective communication initialization and model training starting. | official/r1/resnet/imagenet_main.py |
| run_imagenet() | Model training entry, for input function selection and training result return. | official/r1/resnet/imagenet_main.py |
| resnet_main() | Main function for run configuration, training, and validation. | official/r1/resnet/resnet_run_loop.py |

## Distributed Training

1. Import the following header files to the  **official/r1/resnet/resnet_run_loop.py**  file:

    ```python
    from npu_bridge.estimator import npu_ops
    from tensorflow.core.protobuf import rewriter_config_pb2
    ```

2. Initialize collective communication before training.

    Code location:  **main\(\)**  in  **official/r1/resnet/imagenet_main.py**. The modifications are as follows.

    ```python
    def main(_):
        ############## NPU modify begin #############
        # Call the HCCL API to initialize the NPU.
        # Add the following content to the code.
        init_sess, npu_init = resnet_run_loop.init_npu()
        init_sess.run(npu_init)
        ############## NPU modify end ###############

        with logger.benchmark_context(flags.FLAGS):
            run_imagenet(flags.FLAGS)
    ```

3. Define the collective communication initialization function.

    Code location:  **init_npu\(\)**  added in  **official/r1/resnet/resnet_run_loop.py**.

    ```python
    def resnet_main(flags_obj, model_function, input_function, dataset_name, shape=None):...
    ############## NPU modify begin #############
    #Add the following code.
    def init_npu():
    This API is used to manually initialize the NPU.
    Returns:
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

4. Destroy the device allocations after a single training or verification process is complete.

    Code location:  **resnet_main\(\)**  in  **official/r1/resnet/resnet_run_loop.py**. The modified part is the content between  **NPU modify begin**  and  **NPU modify end**.

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
          # When a single training process is complete, destroy the NPU allocations. You need to reinitialize the NPU before starting a new training process so that the HCCL API is available in the new training process:
        # Add the following content to the code.
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
          # When a single training process is complete, destroy the NPU allocations. You need to reinitialize the NPU before starting a new training process so that the HCCL API is available in the new training process:
        # Add the following content to the code.
          init_sess, npu_init = init_npu()
          npu_shutdown = npu_ops.shutdown_system()
          init_sess.run(npu_shutdown)
          init_sess.run(npu_init)
          ############## NPU modify end ###############
    ```

5. Destroy device allocations after training or validation is complete.

    After training or validation is complete, call  **npu_ops.shutdown_system**  to clean up the NPU allocations.

    Code location:  **resnet_main\(\)**  in  **official/r1/resnet/resnet_run_loop.py**. The modifications are as follows.

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
        # After training or validation is complete, destroy the NPU allocations through the npu_ops.shutdown_system API. Add the following content to the code.
        npu_shutdown = npu_ops.shutdown_system()
        init_sess.run(npu_shutdown)
        ############## NPU modify end ###############

        stats = {}
        stats['eval_results'] = eval_results
        stats['train_hooks'] = train_hooks

        return stats
    ```

## Supplementary Information \(Performance Tuning\)

Set the default value of loss scale.

Code location:  **define_imagenet_flags\(\)**  in  **official/r1/resnet/imagenet_main.py**. The modifications are as follows.

```python
def define_imagenet_flags():
    resnet_run_loop.define_resnet_flags(
        resnet_size_choices=['18', '34', '50', '101', '152', '200'],
        dynamic_loss_scale=True,
        fp16_implementation=True)
    flags.adopt_module_key_flags(resnet_run_loop)
    flags_core.set_defaults(train_epochs=90)

    ############## NPU modify begin #############
 # The Ascend AI Processor supports mixed precision training by default. If the value of loss_scale is too large, the gradient may explode. If the value is too small, the gradient may vanish.
# Set loss_scale as follows to avoid the preceding issues.
    flags_core.set_defaults(loss_scale='512')
    ############## NPU modify end ###############
```
