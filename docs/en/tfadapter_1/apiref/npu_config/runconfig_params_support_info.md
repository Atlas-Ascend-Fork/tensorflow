# Supported RunConfig Parameters

This section describes the support for native  **RunConfig**  parameters of TensorFlow in the  **NPURunConfig**  class.

## Parameters Supported by NPURunConfig

- **model_dir**: path for saving the model. Defaults to  **None**.

    If  **model_dir**  set in  **NPURunConfig**  is different from that in  **NPUEstimator**, an error is reported.

    If either  **NPURunConfig**  or  **NPUEstimator**  is configured with  **model_dir**, the configured path applies.

    If neither  **NPURunConfig**  nor  **NPUEstimator**  is configured with  **model_dir**, a  **model_dir__xxxxxxxxxx_**  directory is created in the current script execution path to save the model file.

- **tf_random_seed**: seed of the initialization variable. Defaults to  **None**.
- **save_summary_steps**: interval (in steps) for saving the summary. Defaults to  **0**.

    Applies only to the scenario where  **iterations_per_loop = 1**. If  **iterations_per_loop \> 1**, the configured value may not be saved. For details about how to save information, see "Log and Summary Operators."

- **save_checkpoints_steps**: interval (in steps) for saving the checkpoints. Defaults to  **None**.

  - This parameter is mutually exclusive with  **save_checkpoints_secs**.
  - If  **save_checkpoints_steps**  and  **save_checkpoints_secs**  are set to  **None**, the checkpoints are saved every 100 steps.
  - If the value of  **iterations_per_loop**  is greater than 1, set  **save_checkpoints_steps**  to a positive integer multiple of  **iterations_per_loop**. Failure to do so may lead to checkpoint data not saved as defined by  **save_checkpoints_steps**.

    To save the checkpoint data on only a specific device, modify the training script as follows:

    Original TensorFlow code:

    ```python
    self._classifier=tf.estimator.Estimator(
      model_fn=cnn_model_fn,
      model_dir=self._model_dir,
      config=tf.estimator.RunConfig(
          save_checkpoints_steps=50 if hvd.rank() == 0 else None,
          keep_checkpoint_max=1))
    ```

    Code after porting:

    ```python
    self._classifier=NPUEstimator(
      model_fn=cnn_model_fn,
      model_dir=self._model_dir,
      config=NPURunConfig(
          save_checkpoints_steps=50 if get_rank_id() == 0 else 0,
          keep_checkpoint_max=1))
    ```

- **save_checkpoints_secs**: interval \(in seconds\) for saving the checkpoints. Defaults to  **None**.

    This parameter is mutually exclusive with  **save_checkpoints_steps**.

- **session_config**:  **ConfigProto**  object of session configuration. Defaults to  **None**.
- **keep_checkpoint_max**: maximum number of checkpoint files that can be stored. Defaults to  **5**.
- **keep_checkpoint_every_n_hours**: checkpoint file saving duration in hours. Defaults to  **10000**. This function can be disabled.

    To use this function, set  **keep_checkpoint_max**  to a large value.

- **log_step_count_steps**: interval \(in steps\) for recording the global_step and loss values. Defaults to  **100**.

    Applies only to the scenario where  **iterations_per_loop = 1**. If  **iterations_per_loop \> 1**, the configured value may not be saved. For details about how to save information, see "Log and Summary Operators."

## Parameters Not Supported by NPURunConfig

The following parameters in  **RunConfig**  are not supported in  **NPURunConfig**.

- **train_distribute**: distributed training enable. The distributed configuration is specified by  **experimental_distribute**.

    This parameter is used only by TensorFlow Adapter. You are advised not to set it.

- **device_fn**: function of the  **Device**  field of each operation.
- **protocol**: \(optional\) protocol used to start the server. If the parameter is empty, the gRPC is used by default.
- **eval_distribute**: distributed evaluation enable. The distributed configuration is specified by  **experimental_distribute**.

    This parameter is used only by TensorFlow Adapter. You are advised not to set it.

- **experimental_distribute**: distributed configuration.
