# Sample Reference

## Sample Code

To perform online inference, the model needs to be built and optimized when  **sess.run\(\)**  is executed for the first time, which increases the time consumption. Therefore, try to minimize the initialization frequency across the app lifetime in the implementation. In this sample, the inference workflow is encapsulated in a  **Classifier**  object so that the app can manage the inference process by controlling the lifetime of the  **Classifier**  object.

**infer_from_pb.py**  sample code

```python
# Load an already-trained .pb model to perform inference.
import tensorflow as tf
import os
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge
import time
import numpy as np

def parse_args():
    '''
    Set the model path, input, and output.
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
    Read image information from the tag file.
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
    Normalize input images.
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
    Preprocess input images.
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

        # Configurations of model compilation and tuning on the AI processor.
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # Configuration 1: Schedule the inference job to the AI processor.
        custom_op.parameter_map["use_off_line"].b = True
       # Configuration 2: In the online inference scenario, you are advised to retain the default precision force_fp16 to achieve better performance.
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        # Configuration 3: Select the graph run mode. Set this parameter to 0 in the inference scenario or retain the default value 1 in the training scenario.
        custom_op.parameter_map["graph_run_mode"].i = 0
        # Configuration 4: Disable remapping and MemoryOptimizer.
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # Load the model and set the input and output nodes of the model.
        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        # Model compilation is triggered during the first sess.run(), which takes a long time. You can bind the session instance to its lifecycle.
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        Load a static graph.
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, batch_data):
        """
        Start inference.
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
        Batch processing.
        """
        # Obtain the batch information of the current input data and automatically adjust the data to a fixed batch.
        n_dim = image_data.shape[0]
        batch_size = self.batch_size

        # If the data is insufficient for the entire batch, pad the data.
        m = n_dim % batch_size
        if m < batch_size and m > 0:
            # The part without data is padded with zeros.
            pad = np.zeros((batch_size - m, 224, 224, 3)).astype(np.float32)
            image_data = np.concatenate((image_data, pad), axis=0)

        # Define the minimum batch that can be divided.
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
    # Data preprocessing.
    tf.reset_default_graph()
    print("########NOW Start Preprocess#########")
    images, labels, images_count = image_process(args.image_path, args.label_file)
    # Batch processing.
    print("########NOW Start Batch#########")
    classifier = Classifier()
    batch_images, batch_labels= classifier.batch_process(images, labels)
    # Start inference.
    print("########NOW Start inference#########")
    batch_logits, total_time = classifier.do_infer(batch_images)
    # Compute the accuracy.
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

## Sample Execution

The following uses the ResNet-50 model as an example to describe how to perform online inference.

1. Download a pre-trained model.

    1. Click  [resnet50_for_TensorFlow](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/resnet50_imagenet_dynamic_hw-python/resnet50_tensorflow_1.7.pb)  to download the trained original model file  **resnet50_tensorflow_1.7.pb**.
    2. Prepare the sample dataset ImageNet2012. Download it from the ImageNet official website at  [https://www.image-net.org/](https://www.image-net.org/).

    > [!NOTE]NOTE
    > The downloaded pre-trained model only supports inference with a batch size of 1.You may also convert and freeze the .ckpt model file generated during training into a .pb file with a custom batch size based on your actual requirements.

2. Edit the inference script.

    Create a model script file  **infer_from_pb.py**  and write code by referring to  [Sample Code](#sample-code).

3. Configure environment variables on which the online inference process depends.

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    # Add the path of the current script to PYTHONPATH. For example:
    export PYTHONPATH="$PYTHONPATH:/root/models"
    # Specify the job ID.
    export JOB_ID=10087       # User-defined job ID. Only letters, digits, hyphens (-), and underscores (_) are supported. You are advised not to use a numeric ID starting with 0.
    ```

4. Run inference.

    ```bash
    python3 infer_from_pb.py --model_path=./resnet50_tensorflow_1.7.pb  --image_path=/data/dataset/imagenet2012/val  --label_file=/data/dataset/imagenet2012/val_label.txt  --input_tensor_name=Placeholder:0 --output_tensor_name=fp32_vars/dense/BiasAdd:0
    ```

    > [!NOTE]NOTE
    >The preceding is an example input. You can modify the input parameters as required. For details about how to determine the input/output node names of a .pb model, see  [Reading Node Names from a PB Model File](../common_operation/read_pb_model_node_name.md).
    >
    >When measuring online inference performance, start timing from the second iteration because the first iteration involves operator and graph compilation, which is time-consuming.
