# 模型固化

## sess.run模式下模型转换与保存

sess.run模式下，TensorFlow在训练过程中，通常使用saver = tf.train.Saver\(\)和saver.save\(\)保存模型，一次saver.save\(\)后会生成如下文件：

- checkpoint：文本文件，记录了保存的最新的Checkpoint文件以及其它Checkpoint文件列表。
- model.ckpt.data-00000-of-00001：保存当前参数值即权重。
- model.ckpt.index：保存当前参数名。
- model.ckpt.meta：保存当前图结构。

这种模型权重数据和模型结构是分开保存的方式，在推理场景下，一般使用TensorFlow提供的freeze_graph函数，将权重数据和模型结构合并为pb格式的文件，对应下图虚框所示部分。

![](figures/save_model.png)

TensorFlow提供的freeze_graph函数生成pb文件的主要过程为：

1. 指定网络模型和Checkpoint文件路径。
2. 定义输入节点。对训练而言，例如输入节点为IteratorV2，而推理需要的输入为placeholder。
3. 定义输出节点。对训练而言，输出节点为loss值，而推理需要的输出通常为loss前面的节点，例如Argmax或BiasAdd等。
4. 通常情况下，训练图和推理图中对同一个算子处理方式不同（例如BatchNorm和dropout等算子），因此需要调用网络模型生成推理图。

    - 对于BatchNorm算子：在训练时BatchNorm算子的平均值和方差在训练时由训练样本进行计算得到，但在推理时，该算子的平均值和方差由样本的滑动平均来计算，因此BatchNorm在训练和推理时需要不同的平均值计算方式。
    - 对于dropout算子：在推理时，需要屏蔽dropout，rate设置为1：

        ```python
            if is_training:
                x = npu_ops.dropout(x, 0.65)
            else:
                x = npu_ops.dropout(x, 1.0)
        ```

    基于以上差异点，用户需要找到训练脚本中推理测试逻辑的入口函数，并在执行时设置is_training=False，从而生成推理图。

    ```python
        # 调用网络模型生成推理图，alexnet.inference为训练脚本中推理测试逻辑的入口函数
        logits = alexnet.inference(inputs, version="he_uniform", num_classes=1000, is_training=False)
    ```

5. 使用tf.train.write_graph将上述推理图保存成pb图文件，作为freeze_graph函数的输入。
6. 使用freeze_graph将tf.train.write_graph生成的pb图文件与Checkpoint文件合并，生成用于推理的pb图文件。

代码示例：

```python
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.npu_init import *

# 导入网络模型文件
import alexnet
# 指定checkpoint路径
ckpt_path = "/opt/npu/model_ckpt/alexnet/model_8p/model.ckpt-0"

def main():
    tf.reset_default_graph()
    # 定义网络的输入节点
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # 调用网络模型生成推理图
    logits = alexnet.inference(inputs, version="he_uniform",
                                  num_classes=1000, is_training=False)
    # 定义网络的输出节点
    predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    with tf.Session() as sess:
        #保存图，在./pb_model文件夹中生成model.pb文件
        # model.pb文件将作为input_graph给到接下来的freeze_graph函数
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')    # 通过write_graph生成模型文件
        freeze_graph.freeze_graph(
                input_graph='./pb_model/model.pb',   # 传入write_graph生成的模型文件
                input_saver='',
                input_binary=False,
                input_checkpoint=ckpt_path,  # 传入训练生成的Checkpoint文件
                output_node_names='output',  # 与定义的推理网络输出节点保持一致
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                output_graph='./pb_model/alexnet.pb',   # 需要生成的推理网络的名称
                clear_devices=False,
                initializer_nodes='')
    print("done")

if __name__ == '__main__':
    main()
```

freeze_graph的关键参数解释，其他参数保持默认：

- input_graph：模型文件，通过write_graph生成的模型文件。
- input_binary：配合input_graph用，为True时，input_graph为二进制，为False时，input_graph为文件。默认值是False。
- input_checkpoint：Checkpoint文件地址。
- output_node_names：输出节点的名字，有多个时用逗号分开。
- output_graph：用来保存转换后的模型输出文件，即pb文件的保存地址。

运行之后，./pb_model/文件夹中就会出现alexnet.pb文件，这是我们转换后的用于推理的pb图文件。

> [!NOTE]说明
> 依赖的环境变量请参考[执行单Device训练](../migration/model_training/single_device_training.md)。

## Estimator模式下模型转换与保存

Estimator可以保存ckpt和saved_model两种格式的模型。ckpt方式与session.run相似，建议保存为saved_model格式，可以轻量化保存模型，同时避免一些可能出现的错误。saved_model模型一般通过estimator.export_savedmodel保存，通常包含如下几个部分：

```text
|--- saved_model.pb     # 保存网络结构
|--- variables         # 参数权重，包含了所有模型的变量（tf.variable objects）参数
|---|--- variables.data-00000-of-00001
|---|--- variables.index
```

如果需要将saved_model模型转换成用于推理的pb模型，主要思路如下：

1. 定义输入节点。

    Estimator在训练时接受的输入是Iterator格式，方便epoch之间迭代，保存推理用模型前要用placeholder定义一个具体输入。

    ```python
    def serving_input_fn():
      input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
      input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
          'input_ids': input_ids,
      })
      return input_fn
    ```

2. 保存saved_model模型。

    Estimator保存模型可以直接调用export_savedmodel函数，从而自动完成模式切换并固定graph。

    ```python
    if FLAGS.do_export:
      estimator.evaluate()
      estimator.export_savedmodel(FLAGS.output_dir, serving_input_fn)
    ```

3. 冻结pb模型。

    通过TensorFlow的freeze_graph函数直接冻结成pb，注意如果使用了NPU自定义算子，需要在freeze_graph源码中加入自定义模块的导入代码。

    ```python
    import tensorflow as tf
    from tensorflow.python.tools import freeze_graph
    from npu_bridge.npu_init import *

    freeze_graph.freeze_graph(
                    input_saved_model_dir='savedModel',
                    output_node_names='output',      # 与定义的推理网络输出节点保持一致
                    output_graph='test.pb',          # 需要生成的推理网络的名称
                            initializer_nodes='',
                    input_graph= None,
                    input_saver= False,
                    input_binary=False,
                    input_checkpoint=None,
                    restore_op_name=None,
                    filename_tensor_name=None,
                    clear_devices=False,
                    input_meta_graph=False)
    ```
