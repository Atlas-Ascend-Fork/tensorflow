# 样例参考

## 样例代码

使用在线推理需要考虑到sess.run首次执行时需要对模型进行编译和优化，耗时会增多。在编写推理应用时，应尽量保证应用生命周期内不频繁初始化。本例中，我们将推理过程封装到Classifier类中，以便应用可以控制Classifier对象的生命周期。

样例代码infer_from_pb.py：

```python
# 通过加载已经训练好的pb模型，执行推理
import tensorflow as tf
import os
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge
import time
import numpy as np

def parse_args():
    '''
    用户自定义模型路径、输入、输出
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', default=1,
                        help="""batchsize""")
    parser.add_argument('--model_path', default='pb/resnet50HC.pb',
                        help="""pb path""")
    parser.add_argument('--image_path', default='image-50000',
                        help="""the data path""")
    parser.add_argument('--label_file', default='val_label.txt',
                        help="""label file""")
    parser.add_argument('--input_tensor_name', default='input_data:0',
                        help="""input_tensor_name""")
    parser.add_argument('--output_tensor_name', default='resnet_model/final_dense:0',
                        help="""output_tensor_name""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

def read_file(image_name, path):
    '''
    从标签文件中读取图片的相关信息
    '''
    with open(path, 'r') as cs:
        rs_list = cs.readlines()
        for name in rs_list:
            if image_name in str(name):
                num = str(name).split(" ")[1]
                break
    return int(num) + 1

def normalize(inputs):
    '''
    图像归一化
    '''
    mean = [121.0, 115.0, 100.0]
    std =  [70.0, 68.0, 71.0]
    mean = tf.expand_dims(tf.expand_dims(mean, 0), 0)
    std = tf.expand_dims(tf.expand_dims(std, 0), 0)
    inputs = inputs - mean
    inputs = inputs * (1.0 / std)
    return inputs

def image_process(image_path, label_file):
    '''
    对输入图像进行一定的预处理
    '''
    imagelist = []
    labellist = []
    images_count = 0
    for file in os.listdir(image_path):
        with tf.Session().as_default():
            image_file = os.path.join(image_path, file)
            image_name = image_file.split('/')[-1].split('.')[0]
            #images preprocessing
            image= tf.gfile.FastGFile(image_file, 'rb').read()
            img = tf.image.decode_jpeg(image, channels=3)
            bbox = tf.constant([0.1,0.1,0.9,0.9])
            img = tf.image.crop_and_resize(img[None, :, :, :], bbox[None, :], [0], [224, 224])[0]
            img = tf.clip_by_value(img, 0., 255.)
            img = normalize(img)
            img = tf.cast(img, tf.float16)
            images_count = images_count + 1
            img = img.eval()
            imagelist.append(img)
            tf.reset_default_graph()

            # read image label from label_file
            label = read_file(image_name, label_file)
            labellist.append(label)

    return np.array(imagelist), np.array(labellist),images_count


class Classifier(object):
    #set batchsize:
    args = parse_args()
    batch_size = int(args.batchsize)

    def __init__(self):

        # AI处理器模型编译和优化配置
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 配置1： 选择在AI处理器上执行推理
        custom_op.parameter_map["use_off_line"].b = True
        # 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        # 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
        custom_op.parameter_map["graph_run_mode"].i = 0
        # 配置4：关闭remapping和MemoryOptimizer
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # 加载模型，并指定该模型的输入和输出节点
        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        加载静态图
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, batch_data):
        """
        执行推理
        """
        out_list = []
        total_time = 0
        i = 0
        for data in batch_data:
            t = time.time()
            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: data})
            if i > 0:
                total_time = total_time + time.time() - t
            i = i + 1
            out_list.append(out)
        return np.array(out_list), total_time

    def batch_process(self, image_data, label_data):
        """
        批处理
        """
        # 获取当前输入数据的批次信息，自动将数据调整为固定批次
        n_dim = image_data.shape[0]
        batch_size = self.batch_size

        # 如果数据不足以用于整个批次，则需要进行数据补齐
        m = n_dim % batch_size
        if m < batch_size and m > 0:
            # 不足部分的数据以0进行填充补齐
            pad = np.zeros((batch_size - m, 224, 224, 3)).astype(np.float32)
            image_data = np.concatenate((image_data, pad), axis=0)

        # 定义可以被分成的最小批次
        mini_batch = []
        mini_label = []
        i = 0
        while i < n_dim:
            # Define the mini batches that can be divided into several batches
            mini_batch.append(image_data[i: i + batch_size, :, :, :])
            mini_label.append(label_data[i: i + batch_size])
            i += batch_size

        return mini_batch, mini_label

def main():
    args = parse_args()
    top1_count = 0
    top5_count = 0
    # 数据预处理
    tf.reset_default_graph()
    print("########NOW Start Preprocess#########")
    images, labels, images_count = image_process(args.image_path, args.label_file)
    # 批处理
    print("########NOW Start Batch#########")
    classifier = Classifier()
    batch_images, batch_labels= classifier.batch_process(images, labels)
    # 开始执行推理
    print("########NOW Start inference#########")
    batch_logits, total_time = classifier.do_infer(batch_images)
    # 计算精度
    batchsize = int(args.batchsize)
    total_step = int(images_count / batchsize)
    print("########NOW Start Compute Accuracy!!!#########")
    for i in range(total_step):
        top1acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(batch_logits[i], 1), batch_labels[i]), tf.float32))
        top5acc = tf.reduce_sum(tf.cast(tf.nn.in_top_k(batch_logits[i], batch_labels[i], 5), tf.float32))
        with tf.Session().as_default():
            tf.reset_default_graph()
            top1_count += top1acc.eval()
            top5_count += top5acc.eval()

    print('+----------------------------------------+')
    print('the correct num is {}, total num is {}.'.format(top1_count, total_step * batchsize))
    print('Top1 accuracy:', top1_count / (total_step * batchsize) * 100)
    print('Top5 accuracy:', top5_count / (total_step * batchsize) * 100)
    print('images number = ', total_step * batchsize)
    print('images/sec = ', (total_step * batchsize) / total_time)
    print('+----------------------------------------+')

if __name__ == '__main__':
    main()
```

## 样例执行

以ResNet50模型为例，执行在线推理样例。

1. 下载预训练模型。

    1. 在Ascend Gitee仓 \> ModelZoo-TensorFlow的[resnet50_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/resnet50_for_TensorFlow)路径下，参考README下载已经训练好的原始模型文件“resnet50_tensorflow_1.7.pb”。
    2. 准备好样例数据集ImageNet2012，您可以从ImageNet官方网站[https://www.image-net.org/](https://www.image-net.org/)获取数据集。

    > [!NOTE]说明
    > 下载的预训练模型文件只支持输入batchsize为1的推理场景，用户也可根据实际，将训练产生的“.ckpt“模型文件转换冻结为自定义batchsize的“.pb“文件。

2. 编辑推理脚本。

    创建“infer_from_pb.py“模型脚本文件，并参考[样例代码](#样例代码)写入相关代码。

3. 配置在线推理进程依赖的环境变量。

    ```bash
    # 配置CANN软件环境变量，以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # 添加当前脚本所在路径到PYTHONPATH，例如：
    export PYTHONPATH="$PYTHONPATH:/root/models"
    # 指定任务ID
    export JOB_ID=10087       # 任务ID，用户自定义，仅支持大小写字母，数字，中划线，下划线。不建议使用以0开头的纯数字
    ```

4. 执行推理。

    ```bash
    python3 infer_from_pb.py --model_path=./resnet50_tensorflow_1.7.pb  --image_path=/data/dataset/imagenet2012/val  --label_file=/data/dataset/imagenet2012/val_label.txt  --input_tensor_name=Placeholder:0 --output_tensor_name=fp32_vars/dense/BiasAdd:0
    ```

    > [!NOTE]说明
    > 上述为样例输入，用户可根据实际修改传入参数。用户在不确定pb模型文件节点名时，可参考[读取pb模型文件的节点名称](../common_operation/read_pb_model_node_name.md)获取模型的输入输出节点名。
    >
    > 在计算在线推理性能时，由于首轮推理会进行算子和图编译，耗时较长，因此从第二轮开始计时。
