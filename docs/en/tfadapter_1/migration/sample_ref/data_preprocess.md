# Data Preprocessing

The data preprocessing process is the same as that of the original model. Some of the code is tailored to the  AI processor  for higher compute capability. The displayed code shows the modifications.

## Defining the Input Function input_fn

Data preprocessing of the ImageNet dataset is used as an example. The modified .py files and functions for adapting to the  AI processor  are as follows.

| Function | Description | Location |
| --- | --- | --- |
| input_fn() | Input function that processes the dataset for Estimator training and outputs real data. | official/r1/resnet/imagenet_main.py |
| resnet_main() | Main API that contains data input, run configuration, training, and validation. | official/r1/resnet/resnet_run_loop.py |

1. Import the following header files to the  **official/r1/resnet/imagenet_main.py**  file:

    ```bash
    from hccl.manage.api import get_rank_size
    from hccl.manage.api import get_rank_id
    ```

2. During data reading, obtain the  AI processor  quantity and IDs to support data parallel.

    Code location:  **input_fn\(\)**  in  **official/r1/resnet/imagenet_main.py**. The modified part is the content between  **NPU modify begin**  and  **NPU modify end**.

    ```python
    def input_fn(is_training, data_dir, batch_size, num_epochs=1, dtype=tf.float32,
                  datasets_num_private_threads=None, parse_record_fn=parse_record,
                  input_context=None, drop_remainder=False, tf_data_experimental_slack=False):
        """Function that provides training and validation batches.
         Parameter description:
           is_training: boolean value indicating whether the input is used for training.
           data_dir: file path that contains the input dataset.
           batch_size: size of each batch.
           num_epochs: number of epochs.
           dtype: data type of an image or feature.
           datasets_num_private_threads: number of threads dedicated to tf.data.
           parse_record_fn: entry point function for parsing TFRecords.
           input_context: tf.distribute.InputContext object passed by tf.distribute.Strategy
           drop_remainder: specifies whether to retain or discard the last batch if the data volume of the last batch is smaller than the value of batch_size. If it is set to True, the batch dimension is fixed.
           tf_data_experimental_slack: specifies whether to enable the experimental_slack option of tf.data.

         Returns:
           A dataset that can be used for iteration.
         """
         # Obtain the file path.
         filenames = get_filenames(is_training, data_dir)
         # Split the file based on the first dimension.
         dataset = tf.data.Dataset.from_tensor_slices(filenames)
         if input_context:
             # Obtain the AI processor quantity and IDs to support data parallel.
             ############## NPU modify begin #############
             dataset = dataset.shard(get_rank_size(),get_rank_id())
             ############## NPU modify end ###############

             # tf.compat.v1.logging.info(
             #     'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
             #         input_context.input_pipeline_id, input_context.num_input_pipelines))
             # dataset = dataset.shard(input_context.num_input_pipelines,
             #                         input_context.input_pipeline_id)

         if is_training:
             # Randomize the files.
             dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

         # cycle_length = 10 Read and deserialize 10 files in parallel. You can increase the value if the CPU resources are sufficient.
         dataset = dataset.interleave(
             tf.data.TFRecordDataset,
             cycle_length=10,
             num_parallel_calls=tf.data.experimental.AUTOTUNE)

         return resnet_run_loop.process_record_dataset(
             dataset=dataset,
             is_training=is_training,
             batch_size=batch_size,
             shuffle_buffer=_SHUFFLE_BUFFER,
             parse_record_fn=parse_record_fn,
             num_epochs=num_epochs,
             dtype=dtype,
             datasets_num_private_threads=datasets_num_private_threads,
             drop_remainder=drop_remainder,
             tf_data_experimental_slack=tf_data_experimental_slack,
         )
    ```

3. In  **input_fn\(\)**  in the training or testing scenario,  **drop_remainder**  must be set to  **True**.

    Code location:  **resnet_main\(\)**  in  **official/r1/resnet/resnet_run_loop.py**. The modified parts are the  **input_fn_train\(\)**  and  **input_fn_eval\(\)**  subfunctions.

    ```python
      def input_fn_train(num_epochs, input_context=None):
        ############## NPU modify #############
        # Set dtype to tf.float16 for better data transfer performance.
        # In the current version, drop_remainder can only be set to True.
        # batch_size indicates the batch size of a single device instead of the global batch size.
        return input_function(
            is_training=True,
            data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=num_epochs,
            dtype=tf.float16,
            input_context=input_context,
            drop_remainder=True)

      def input_fn_eval():
        # Set dtype to tf.float16 for better data transfer performance.
        # In the current version, drop_remainder can only be set to True.
        # batch_size indicates the batch size of a single device instead of the global batch size.
         return input_function(
             is_training=False,
             data_dir=flags_obj.data_dir,
             batch_size=flags_obj.batch_size,
             num_epochs=1,
             dtype=tf.float16,
             input_context=True,
             drop_remainder=True)
      ############## NPU modify end ###############

      # input_fn() for training and validation in the code are as follows.
      # def input_fn_train(num_epochs, input_context=None):
      #     return input_function(
      #         is_training=True,
      #         data_dir=flags_obj.data_dir,
      #         batch_size=distribution_utils.per_replica_batch_size(
      #             flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
      #         num_epochs=num_epochs,
      #         dtype=flags_core.get_tf_dtype(flags_obj),
      #         datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      #         input_context=input_context)
      #
      # def input_fn_eval():
      #     return input_function(
      #         is_training=False,
      #         data_dir=flags_obj.data_dir,
      #         batch_size=distribution_utils.per_replica_batch_size(
      #             flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
      #         num_epochs=1,
      #         dtype=flags_core.get_tf_dtype(flags_obj))
    ```
