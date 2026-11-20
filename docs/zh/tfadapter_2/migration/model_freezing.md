# 模型固化

TensorFlow保存图graph、权重weights的过程称为freezing，在保存过程中会产生一个protobuf文件，简称pb文件。导出并保存pb文件的方法使用的是原生TensorFlow框架的能力，下面仅简要给出一些步骤示例。

1. 保存SavedModel模型。

    ```python
    tf.saved_model.save(network, "save_path")
    ```

    在save_path文件夹下会生成如下文件/文件夹：

    ```text
    |--- save_model.pb     # 保存网络结构
    |--- variables         # 权重参数存储目录
    |--- assets            # 所需的外部文件存储目录，例如初始化的词汇表文件
    ```

2. 冻结pb模型。

    加载上述步骤导出的SavedModel模型，并将SavedModel模型冻结为带权重的pb模型，代码示例如下。

    ```python
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from tensorflow.python.framework import graph_util

    # 加载SavedModel模型
    saved_model_dir = "save_path"
    model = tf.saved_model.load(saved_model_dir)
    # 初始化signatures
    infer = model.signatures["serving_default"]
    # 冻结带权重的pb文件
    frozen_func = convert_variables_to_constants_v2(infer)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                         logdir="./",
                         name="frozen_graph.pb",
                         as_text=False)
    ```
