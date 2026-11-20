# 运行配置

模型配置函数主要在resnet_main\(\)中，其函数介绍如下：

| 接口 | 简介 | 位置 |
|------|------|------|
| resnet_main() | 包含运行配置、训练以及验证的主要函数。 | `official/r1/resnet/resnet_run_loop.py` |

1. 在“official/r1/resnet/resnet_run_loop.py“增加以下头文件：

    ```python
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    ```

2. 通过NPURunConfig替代RunConfig来配置运行参数。

    代码位置：“official/r1/resnet/resnet_run_loop.py“的resnet_main\(\)函数：

    ```python
      ############## NPU modify begin #############
      # 使用NPURunConfig替换RunConfig，适配昇腾AI处理器，每115200步保存一次checkpoint，每10000次保存一次summary，
      # 对数据进行预处理，使用混合精度模式提升训练速度。
      run_config = NPURunConfig(
          model_dir=flags_obj.model_dir,
          session_config=session_config,
          save_checkpoints_steps=115200,
          enable_data_pre_proc=True,
          iterations_per_loop=100,
          # enable_auto_mix_precision=True,
          # 精度模式设置为混合精度模式。
          precision_mode='allow_mix_precision',
          hcom_parallel=True
      )
      ############## NPU modify end ###############

      # 原代码中运行参数配置如下：
      # run_config = tf.estimator.RunConfig(
      #     train_distribute=distribution_strategy,
      #     session_config=session_config,
      #     save_checkpoints_secs=60 * 60 * 24,
      #     save_checkpoints_steps=None)
    ```

    > [!NOTE]说明
    > 关于设置混合精度模式：precision_mode='allow_mix_precision，具体可参考[设置混合精度模式](../performance_tuning/mixed_precision_training.md#设置混合精度模式)。

3. 创建NPUEstimator，使用NPUEstimator接口代替tf.estimator.Estimator。

    代码位置：“official/r1/resnet/resnet_run_loop.py“的resnet_main\(\)函数（修改部分如下）：

    ```python
        # 使用`NPUEstimator`接口代替tf.estimator.Estimator
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
        # 原代码中创建Estimator如下：
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
