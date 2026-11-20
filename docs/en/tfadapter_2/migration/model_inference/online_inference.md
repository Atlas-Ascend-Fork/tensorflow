# Online Inference

## Introduction

Online inference refers to real-time inference using a trained .pb model in TensorFlow. Compared with offline inference, online inference has higher timeliness requirements and is usually used for data center inference.

By referring to this section, you will be able to easily port TensorFlow-based inference applications to the Ascend platform.

## Porting Sample

This section takes the ResNet-50 model as an example to show how to port a TensorFlow 2.6.5 online inference application that runs in the CPU/GPU environment to the  AI processor.

### Preparation

Prepare the TensorFlow 2.6.5 online inference code and datasets, and make sure the code can be run properly on the CPU/GPU.

### Online Inference on the CPU/GPU

The TensorFlow 2.6.5 online inference code mainly includes the following actions:

1. Prepare the ResNet-50.pb model, input node, output node, and dataset.
2. Call  **sess.run\(\)**  to perform inference. The  **feed_dict**  in  **sess.run\(\)**  is used to assign values to the tensor created using  **placeholder**. The feed \(input\) data can be used as the parameter called by  **run\(\)**.

The key inference code is as follows:

```python
def load_graph(frozen_graph):
    with tf.io.gfile.GFile(frozen_graph,"rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="")
    return graph

def NetworkRun(modelPath,inputPath,outputPath):
    graph = load_graph(modelPath)
    input_nodes = graph.get_tensor_by_name('Input:0')
    output_nodes = graph.get_tensor_by_name('Identity:0')

    with tf.compat.v1.Session(graph=graph) as sess:
        files = os.listdir(inputPath)
        files.sort()
        for file in files:
            if file.endswith(".bin"):
                input_img = np.fromfile(inputPath+"/"+file,dtype="float32").reshape(1,224,224,3)
                t0 = time.time()
                out = sess.run(output_nodes, feed_dict= {input_nodes: input_img,})
                t1 = time.time()
                out.tofile(outputPath+"/"+"cpu_out_"+file)
                print("{}, Inference time: {:.3f} ms".format(file,(t1-t0)*1000))
```

## Porting the Inference Script to the NPU

1. Import the NPU-related configuration library.

   ```python
   import npu_device
   from npu_device.compat.v1.npu_init import *
   npu_device.compat.enable_v1()
   from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
   ```

2. Add the NPU-related configuration before  **sess.run**.

   ```python
   config_proto = tf.compat.v1.ConfigProto()
   custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
   custom_op.name = "NpuOptimizer"
   custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
   config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
   tf_config = npu_config_proto(config_proto=config_proto)
   ```

### Checking the Porting Result

In the ported online inference script on the  AI processor, the execution success flag is the same as the training success flag. You can tell in the following scenarios:

1. The keywords  **tf_adapter**  and message "The model has been compiled on the Ascend AI processor" are printed.

    ![](../figures/inference_result.png)

2. The computational graph dump can be generated after you can enable  **DUMP_GE_GRAPH**.

### Checking the Inference Performance and Accuracy

The inference performance is computed based on the time difference before and after  **sess.run\(\)**  is executed. According to the inference result in this sample, the performance on the NPU is much better than that on the CPU.

![](../figures/inference_result_pefor_precision.png)

The inference accuracy is computed by converting the output .bin file into a .txt file for comparison. According to the inference result in this sample, the accuracy on the NPU is similar to that on the CPU.

![](../figures/inference_compare.png)

## Code Sample

Inference script ported to the  AI processor:

```python
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import tensorflow as tf
import numpy as np
import os
import time
import argparse

# np.random.seed(10)

def load_graph(frozen_graph):
    with tf.io.gfile.GFile(frozen_graph,"rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name="")
    return graph

def NetworkRun(modelPath,inputPath,outputPath):
    graph = load_graph(modelPath)
    input_nodes = graph.get_tensor_by_name('Input:0')
    output_nodes = graph.get_tensor_by_name('Identity:0')
    # Adapt to the NPU.
    config_proto = tf.compat.v1.ConfigProto()
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    tf_config = npu_config_proto(config_proto=config_proto)

    with tf.compat.v1.Session(config=tf_config,graph=graph) as sess:
        files = os.listdir(inputPath)
        files.sort()
        for file in files:
            if file.endswith(".bin"):
                input_img = np.fromfile(inputPath+"/"+file,dtype="float32").reshape(1,224,224,3)
                t0 = time.time()
                out = sess.run(output_nodes, feed_dict= {input_nodes: input_img,})
                print('out---',out)
                t1 = time.time()
                out.tofile(outputPath+"/"+"cpu_out_"+file)
                print("{}, Inference time: {:.3f} ms".format(file,(t1-t0)*1000))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./resnet50_tf2.pb")
    parser.add_argument("--input", type=str, default="./input_bin/")
    parser.add_argument("--output", type=str, default="./npu_output/")
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    NetworkRun(args.model,args.input,args.output)
```
