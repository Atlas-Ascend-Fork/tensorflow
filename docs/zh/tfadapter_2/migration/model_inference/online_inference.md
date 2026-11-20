# 在线推理

## 在线推理简介

在线推理是指在TensorFlow框架内使用已经训练好的pb模型实时进行推理，相比于离线推理场景，在线推理类业务时效性要求较高，常用于数据中心推理场景。

开发者可参考本节方便地将原来基于TensorFlow框架做推理的应用迁移到昇腾平台。

## 迁移示例

下面，我们以ResNet50模型为例，介绍如何将基于TensorFlow 2.6.5框架和CPU/GPU环境运行的在线推理应用迁移到AI处理器上。

### 迁移前准备

准备基于TensorFlow 2.6.5框架的CPU/GPU在线推理代码和数据集，并在CPU/GPU环境正常执行。

### 熟悉CPU/GPU在线推理流程

基于TensorFlow 2.6.5框架的在线推理代码主要包括：

1. 准备resnet50.pb模型、输入节点、输出节点、数据集。
2. 调用sess.run\(\)执行推理。sess.run\(\)中的feed_dict的作用是给使用placeholder创建出来的tensor赋值，我们可以提供feed（输入）数据，作为run\(\)调用的参数。

关键推理代码为：

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

### 推理脚本迁移到NPU

1. 引入NPU相关配置库。

   ```python
   import npu_device
   from npu_device.compat.v1.npu_init import *
   npu_device.compat.enable_v1()
   from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
   ```

2. 在sess.run之前增加NPU的相关config配置。

   ```python
   config_proto = tf.compat.v1.ConfigProto()
   custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
   custom_op.name = "NpuOptimizer"
   custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
   config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
   tf_config = npu_config_proto(config_proto=config_proto)
   ```

### 查看是否迁移成功

在AI处理器执行迁移后的在线推理脚本，其执行成功的标志和训练成功打印一致，方法包括：

1. 出现tf_adapter和The model has been compiled on the Ascend AI processor的关键字打印。

    ![](../figures/inference_result.png)

2. 或者打开Dump计算图开关DUMP_GE_GRAPH，看能否产生Dump计算图。

### 检查推理性能和精度

推理性能是通过sess.run\(\)前后时间打点差值来计算的。从本样例推理结果看，NPU的性能远优于CPU的性能。

![](../figures/inference_result_pefor_precision.png)

推理精度是将输出bin文件转换为txt进行比较，从本样例推理结果可以看出NPU与CPU精度差别不大。

![](../figures/inference_compare.png)

## 完整代码示例

迁移到AI处理器的推理脚本：

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
    # 适配NPU
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
