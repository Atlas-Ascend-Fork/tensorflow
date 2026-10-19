# RunConfig参数支持说明

本节描述了TensorFlow RunConfig原生参数在NPURunConfig类中的支持情况。

## NPURunConfig支持参数

- **model_dir**：保存模型路径。默认值：None。

    如果NPURunConfig和NPUEstimator配置的model_dir不同，系统报错。

    如果NPURunConfig和NPUEstimator仅一个接口配置model_dir，以配置的路径为准。

    如果NPURunConfig和NPUEstimator均未配置model_dir，则系统在当前脚本执行路径创建一个model_dir_xxxxxxxxxx目录保存模型文件。

- **tf_random_seed**：初始化变量的种子。默认值：None。
- **save_summary_steps**：每隔多少step保存一次Summary。默认值：0。

    该参数仅适用于iterations_per_loop=1的场景，iterations_per_loop\>1时，可能无法按照配置的值保存，请参考“Log/Summary”专题实现信息保存。

- **save_checkpoints_steps**：每隔多少step保存一次checkpoint。默认值：None。

  - 与save_checkpoints_secs不能同时配置。
  - 如果save_checkpoints_steps和save_checkpoints_secs都为None，则每隔100个step保存一次checkpoint。
  - iterations_per_loop\>1的场景下，要求save_checkpoints_steps必须大于或等于iterations_per_loop，且是iterations_per_loop的整数倍，否则不会按照save_checkpoints_steps配置的值保存checkpoint数据。

    如果用户只希望在某个Device上保存checkpoint，而不希望在其他Device上保存checkpoint数据，可以按照如下方法修改训练脚本。

    TensorFlow原始代码：

    ```python
    self._classifier=tf.estimator.Estimator(
      model_fn=cnn_model_fn,
      model_dir=self._model_dir,
      config=tf.estimator.RunConfig(
          save_checkpoints_steps=50 if hvd.rank() == 0 else None,
          keep_checkpoint_max=1))
    ```

    迁移后的代码：

    ```python
    self._classifier=NPUEstimator(
      model_fn=cnn_model_fn,
      model_dir=self._model_dir,
      config=NPURunConfig(
          save_checkpoints_steps=50 if get_rank_id() == 0 else 0,
          keep_checkpoint_max=1))
    ```

- **save_checkpoints_secs**：每隔多少秒保存一次checkpoint。默认值：None。

    与save_checkpoints_steps不能同时配置。

- **session_config**：设置session参数的ConfigProto格式对象。默认值：None。
- **keep_checkpoint_max**：最大保存多少个Checkpoint文件。默认值：5。
- **keep_checkpoint_every_n_hours**：保留N个小时的Checkpoint文件。默认值：10000，可有效禁用该功能。

    如需使用该功能，keep_checkpoint_max需要配置足够大。

- **log_step_count_steps**：每隔多少step，记录global_step和loss值一次。默认值：100。

    该参数仅适用于iterations_per_loop=1的场景，iterations_per_loop\>1时，可能无法按照配置的值保存，请参考“Log/Summary”专题实现信息保存。

## NPURunConfig不支持参数

如下RunConfig中的参数在NPURunConfig中不支持。

- **train_distribute**：指明分布式训练策略，分布式相关配置由experimental_distribute指定。

    仅TF Adapter迁移工具会使用该参数，不建议用户单独使用。

- **device_fn**：获取每个Operation的Device字段的function。
- **protocol**：可选参数，指定启动Server时使用的协议。无表示默认为GRPC。
- **eval_distribute**：是否分布式验证，分布式相关配置由experimental_distribute指定。

    仅TF Adapter迁移工具会使用该参数，不建议用户单独使用。

- **experimental_distribute**：分布式配置。
