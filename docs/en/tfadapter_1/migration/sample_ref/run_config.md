# Run Configuration

Set run configuration using the  **resnet_main\(\)**  function.

| Function | Description | Location |
| --- | --- | --- |
| resnet_main() | Main function for run configuration, training, and validation. | official/r1/resnet/resnet_run_loop.py |

1. Import the following header files to the  **official/r1/resnet/resnet_run_loop.py**  file:

    ```python
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    ```

2. Replace  **RunConfig**  with  **NPURunConfig**  to configure run parameters.

    Code location:  **resnet_main\(\)**  in  **official/r1/resnet/resnet_run_loop.py**.

    ```python
      ############## NPU modify begin #############
      # Replace RunConfig with NPURunConfig to adapt to the Ascend AI Processor. Save the checkpoint every 115200 steps and summary every 10000 times.
      # Preprocess data and enable the mixed precision mode to improve the training speed.
      run_config = NPURunConfig(
          model_dir=flags_obj.model_dir,
          session_config=session_config,
          save_checkpoints_steps=115200,
          enable_data_pre_proc=True,
          iterations_per_loop=100,
          # enable_auto_mix_precision=True,
          # Set precision_mode to allow_mix_precision.
          precision_mode='allow_mix_precision',
          hcom_parallel=True
      )
      ############## NPU modify end ###############

        # The run configuration in the code is as follows.
      # run_config = tf.estimator.RunConfig(
      #     train_distribute=distribution_strategy,
      #     session_config=session_config,
      #     save_checkpoints_secs=60 * 60 * 24,
      #     save_checkpoints_steps=None)
    ```

    > [!NOTE]NOTE
    > For details about how to set the mixed precision mode \(**precision_mode='allow_mix_precision'**\), see  [Setting the Mixed Precision Mode](../performance_tuning/mixed_precision_training.md#setting-the-mixed-precision-mode).

3. Create  **NPUEstimator**  to replace  **tf.estimator.Estimator**.

    Code location:  **resnet_main\(\)**  in  **official/r1/resnet/resnet_run_loop.py**. The modifications are as follows.

    ```python
        # Replace tf.estimator.Estimator with NPUEstimator.
        classifier = NPUEstimator(
            model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
            params={
                'resnet_size': int(flags_obj.resnet_size),
                'data_format': flags_obj.data_format,
                'batch_size': flags_obj.batch_size,
                'resnet_version': int(flags_obj.resnet_version),
                'loss_scale': flags_core.get_loss_scale(flags_obj,
                                                        default_for_fp16=128),
                'dtype': flags_core.get_tf_dtype(flags_obj),
                'fine_tune': flags_obj.fine_tune,
                'num_workers': num_workers,
                'num_gpus': flags_core.get_num_gpus(flags_obj),
            })
        # The creation of Estimator in the code is as follows.
        # classifier = tf.estimator.Estimator(
        #     model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
        #     warm_start_from=warm_start_settings, params={
        #         'resnet_size': int(flags_obj.resnet_size),
        #         'data_format': flags_obj.data_format,
        #         'batch_size': flags_obj.batch_size,
        #         'resnet_version': int(flags_obj.resnet_version),
        #         'loss_scale': flags_core.get_loss_scale(flags_obj,
        #                                                 default_for_fp16=128),
        #         'dtype': flags_core.get_tf_dtype(flags_obj),
        #         'fine_tune': flags_obj.fine_tune,
        #         'num_workers': num_workers,
        #     })
    ```
