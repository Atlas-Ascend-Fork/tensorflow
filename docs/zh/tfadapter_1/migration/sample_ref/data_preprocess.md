# 数据预处理

数据预处理流程与原始模型一致，部分位置经改造以适配AI处理器并提升计算性能，展示的示例代码包含改动位置。

## 定义输入函数input_fn

这里以ImageNet数据集数据预处理为例，其数据预处理部分涉及到的适配AI处理器改造的py文件及相关函数接口介绍如下：

| 接口 | 简介 | 位置 |
|------|------|------|
| input_fn() | 输入函数，用于处理数据集用于Estimator训练，输出真实数据。 | `official/r1/resnet/imagenet_main.py` |
| resnet_main() | 包含数据输入、运行配置、训练以及验证的主接口。 | `official/r1/resnet/resnet_run_loop.py` |

1. 在“official/r1/resnet/imagenet_main.py“文件中增加以下头文件：

    ```bash
    from hccl.manage.api import get_rank_size
    from hccl.manage.api import get_rank_id
    ```

2. 在数据读取时，获取AI处理器数量以及AI处理器  ID，用于支持数据并行。

    代码位置：“official/r1/resnet/imagenet_main.py“的input_fn\(\)函数（修改部分为“NPU modify begin”与“NPU modify end ”之间的内容）：

    ```python
    def input_fn(is_training, data_dir, batch_size, num_epochs=1, dtype=tf.float32,
                  datasets_num_private_threads=None, parse_record_fn=parse_record,
                  input_context=None, drop_remainder=False, tf_data_experimental_slack=False):
         """提供训练和验证batches的函数。
         参数解释:
           is_training: 表示输入是否用于训练的布尔值。
           data_dir: 包含输入数据集的文件路径。
           batch_size: 每个batch的大小。
           num_epochs: 数据集的重复数。
           dtype: 图片/特征的数据类型。
           datasets_num_private_threads: tf.data的专用线程数。
           parse_record_fn: 解析tfrecords的入口函数。
           input_context: 由'tf.distribute.Strategy'传入的'tf.distribute.InputContext'对象。
           drop_remainder: 用于标示对于最后一个batch如果数据量达不到batch_size时保留还是抛弃。设置为True,则batch的维度固定。
           tf_data_experimental_slack: 是否启用tf.data的'experimental_slack'选项。

         Returns:
           返回一个可用于迭代的数据集。
         """
         # 获取文件路径
         filenames = get_filenames(is_training, data_dir)
         # 按第一个维度切分文件
         dataset = tf.data.Dataset.from_tensor_slices(filenames)
         if input_context:
             # 获取AI处理器数量以及ID，用于支持数据并行
             ############## NPU modify begin #############
             dataset = dataset.shard(get_rank_size(),get_rank_id())
             ############## NPU modify end ###############

             # tf.compat.v1.logging.info(
             #     'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
             #         input_context.input_pipeline_id, input_context.num_input_pipelines))
             # dataset = dataset.shard(input_context.num_input_pipelines,
             #                         input_context.input_pipeline_id)

         if is_training:
             # 将文件顺序打乱
             dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

         # cycle_length = 10 并行读取并反序列化10个文件，CPU资源充足的场景下可适当增加该值。
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

3. 用于训练和测试的输入函数接口中，drop_remainder需要设置为True。

    代码位置：“/official/r1/resnet/resnet_run_loop.py“中的resnet_main\(\)函数（修改部分为input_fn_train\(\)和input_fn_eval\(\)子函数）：

    ```python
      def input_fn_train(num_epochs, input_context=None):
        ############## NPU modify #############
        # 使用dtype=tf.float16提高数据传输性能。
        # 当前版本的drop_remainder仅支持设置为True。
        # 此处的batch_size指的是单卡的batch大小而不是全局batch大小。
        return input_function(
            is_training=True,
            data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=num_epochs,
            dtype=tf.float16,
            input_context=input_context,
            drop_remainder=True)

      def input_fn_eval():
        # 使用dtype=tf.float16提高数据传输性能
        # 当前版本的drop_remainder只支持为True
        # 这里的batch_size指的是单卡的batch大小而不是全局batch大小
         return input_function(
             is_training=False,
             data_dir=flags_obj.data_dir,
             batch_size=flags_obj.batch_size,
             num_epochs=1,
             dtype=tf.float16,
             input_context=True,
             drop_remainder=True)
      ############## NPU modify end ###############

      # 原代码中用于训练的输入函数接口和用于验证的输入函数接口。
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
