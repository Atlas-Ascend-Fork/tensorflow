# Updating Weights (Online Inference)

## Background

During the inference phase, the training server continuously trains the model to update the weights. However, it would be easier to directly update the latest weights in the inference phase, instead of going through the complete procedure including saving the .pb file, compiling it into an offline model, and then executing the offline model all over. In this scenario, the online inference mode can be used to directly update the weights.

This section describes how to update weights during online inference on TensorFlow.

## Overall Workflow

**Figure  1**  Weight update workflow
![](../figures/weight_update_diagram.png "weight-update-workflow")

As shown in  the figure above, weight update and inference execution can be implemented using loops. If the value of **batch_size** does not change after multiple times of execution, the operations in dashed lines do not need to be performed repeatedly. The main workflow is as follows:

1. Obtain the online inference model and weight information from the .ckpt file, for example. The actual weights to be updated come from the external  _key-value_  pairs.
2. Construct a weight update graph, including the variables and assignment operators to be updated, for example, Assign.
3. Execute the weight update graph to update the  _key-value_  pairs obtained in step 1 to the corresponding weights.
4. Start online inference.

## Samples

```python
import tensorflow as tf
import time
import numpy as np
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops

class TestPolicy(object):
    def __init__(self, ckpt_path):
        # Set the NPU configurations for model compilation and optimization.
        # --------------------------------------------------------------------------------
        config = tf.compat.v1.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # Configuration 1: Perform inference on the Ascend NPU.
        custom_op.parameter_map["use_off_line"].b = True

       # Configuration 2: In the online inference scenario, you are advised to retain the default precision force_fp16 to achieve better performance.
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        # Configuration 3: Disable remapping.
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        # Configuration 4: Set graph_run_mode to inference.
        custom_op.parameter_map["graph_run_mode"].i = 0

        # Configuration 5: Set the AI Core parallelism degree to 4.
        custom_op.parameter_map["stream_max_parallel_num"].s = tf.compat.as_bytes("AIcoreEngine:4")
        # --------------------------------------------------------------------------------

        # Initialize.
        self.sess = tf.compat.v1.Session(config=config)
        self.ckpt_path = ckpt_path
        # Load the model.
        self.load_graph()
        self.graph = self.sess.graph

    def load_graph(self):
        '''
        Load the model from the .ckpt file, obtain the weight information, and construct a weight update graph.
        '''
        saver = tf.compat.v1.train.import_meta_graph(self.ckpt_path + '.meta')
        saver.restore(self.sess, self.ckpt_path)
        self.vars = tf.compat.v1.trainable_variables()
        self.var_placeholder_dict = {}
        self.var_id_to_name = {}
        self.update_op = []
        for id, var in enumerate(self.vars):
            self.var_placeholder_dict[var.name] = tf.compat.v1.placeholder(var.dtype, shape=var.get_shape(), name=("PlaceHolder_" + str(id)))
            self.var_id_to_name[id] = var.name
            self.update_op.append(tf.compat.v1.assign(var, self.var_placeholder_dict[var.name]))
        self.update_op = tf.group(*self.update_op)

        # The actual key-value weight pairs come from the training server. The weight in the .ckpt file is used as an example.
        self.key_value = self.get_dummy_weights_for_test()

    def unload(self):
        '''
        Close the session to destroy allocations.
        '''
        print("====== start to unload ======")
        self.sess.close()

    def get_dummy_weights_for_test(self):
        '''
        Obtain the weight information from the .ckpt file and construct a weight update graph.
        :return: key-value weight pairs
        :NOTES: The actual key-value weight pairs come from the training server. The weight in the .ckpt file is used as an example.
        '''
        weights_data = self.sess.run(self.vars)
        weights_key_value = {}
        for id, var in enumerate(weights_data):
            weights_key_value[self.var_id_to_name[id]] = var
        return weights_key_value

    def get_weights_key_value(self):
        '''
        Obtains the key-value weight pairs.
        :return: key-value weight pairs
        :NOTES: The actual key-value weight pairs come from the training server. The saved weight is used as an example.
        '''
        return self.key_value

    def update_weights(self):
        '''
        Update the weights.
        '''
        feed_dict = {}
        weights_key_value = self.get_weights_key_value()
        for key, weight in weights_key_value.items():
            feed_dict[self.var_placeholder_dict[key]] = weight
        self.sess.run(self.update_op, feed_dict=feed_dict)

    def infer(self, input_image):
        '''
        Start inference.
        :param: input_image. The image data is used in the example.
        :return: output inference result, which is labels in the example
        '''
        image = self.graph.get_operation_by_name('Placeholder').outputs[0]
        label_output = self.graph.get_operation_by_name('accuracy/ArgMax').outputs[0]
        output = self.sess.run([label_output], feed_dict={image: input_image})
        return output


def prepare_input_data(batch):
    '''
    Input data for inference
    :param: batch data batch_size
    :return: inference data
    '''
    image = 255 * np.random.random([batch, 784]).astype('float32')
    return image

if __name__ == "__main__":
    batch_size = 16
    ckpt_path = "./mnist_deep_model/mnist_deep_model"
    policy = TestPolicy(ckpt_path)
    update_count = 10
    for i in range(update_count):
        update_start = time.time()
        policy.update_weights()
        update_consume = time.time() - update_start
        print("Update weight time cost: {} ms".format(update_consume * 1000))
        test_count = 20
        input_data = prepare_input_data(batch_size)
        start_time = time.time()
        for i in range(test_count):
            output = policy.infer(input_data)
            print("result is ", output)
        time_consume = (time.time() - start_time) / (test_count)
        print("Inference average time cost: {} ms \n".format(time_consume * 1000))

    policy.unload()
    print("====== end of test ======")
```
