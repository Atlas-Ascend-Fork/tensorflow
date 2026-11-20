# 样例参考

## 样例代码

在任意目录下创建Client文件夹（本节以$HOME目录为例），用于存放与服务端通信的脚本deploy.py和推理脚本tf_serving_infer.py，目录如下所示：

```text
Client/
 ├── deploy.py
 └── tf_serving_infer.py
```

样例代码deploy.py：

```python
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np
import os
import time

class PredictModelGrpc(object):
    def __init__(self, model_name, input_name, output_name, socket='xxx.xxx.xxx.xxx:8500'):# xxx.xxx.xxx.xxx为服务端IP地址
        self.socket = socket
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.request, self.stub = self.__get_request()
    def __get_request(self):
        channel = grpc.insecure_channel(self.socket, options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                                              ('grpc.max_receive_message_length',
                                                               1024 * 1024 * 1024)])  # 可设置大小
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()

        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"

        return request, stub
    def inference(self, frames):

        t0 = time.time()
        self.request.inputs[self.input_name].CopyFrom(tf.make_tensor_proto(frames, dtype=tf.float32))# 发送请求
        t1 = time.time()
        result = self.stub.Predict.future(self.request, 1000.0) # 执行推理，请求等待时间建议设置为1000.0。
        t2 = time.time()

        res = []
        res.append(tf.make_ndarray(result.result().outputs[self.output_name])[0]) # 获取结果

        t3 = time.time()
        print("Time cost: request.inputs={:.3f} ms, Predict.future={:.3f} ms, get output={:.3f} ms".format((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000))# 耗时打印
        return res
```

样例代码tf_serving_infer.py：

```python
import numpy as np
from PIL import Image
from scipy import misc
import numpy as np
import scipy
import imageio
from deploy import PredictModelGrpc
import time
import sys

data_process_time = []
image_path = sys.argv[1]
# 数据预处理
d0 = time.time()
image = misc.imread(image_path)
resized = scipy.misc.imresize(image, (304, 304, 3)) # 原始.pb模型输入节点的type值，请根据实际值自行修改。
crop_min = abs(304 / 2 - (304 / 2))
crop_max = crop_min + 304
crop_min = int(crop_min)
crop_max = int(crop_max)
image = resized[crop_min:crop_max, crop_min:crop_max, :]

mean_sub = image.astype(np.float32) - np.array([123, 117, 104]).astype(np.float32) # 数据类型从原始.pb模型输入节点的Type中获取，请根据实际的值自行修改。
image = np.expand_dims(np.array(mean_sub), 0)

d1 = time.time()
data_process_time.append(d1 - d0)
model = PredictModelGrpc(model_name='mobileNetv2', input_name='input:0', output_name='MobilenetV2/Logits/output:0') # 根据实际的模型名、模型输入节点名和模型输出节点名自行修改，如果是多输入输出节点时，节点名之间使用;隔开。

# 执行推理
infer_cost_time = []
for i in range(1000):
    t0 = time.time()
    res = model.inference(image)
    t1 = time.time()
    infer_cost_time.append(t1 - t0)
    print("Index= {}, Inference time cost={:.3f} ms".format(i,(t1-t0)*1000))

# 推理耗时打印
print("Batchsize={}, Average inference time cost: {:.3f} ms, Average data process time cost:{:.3f} ms".format(1, (sum(infer_cost_time) - infer_cost_time[0]) / (len(infer_cost_time)) * 1000, (sum(data_process_time) - data_process_time[0]) / (len(data_process_time)) * 1000))
```

> [!NOTE]说明
> 用户在不确定原始.pb模型输入输出节点名以及Type值时，可参考[读取pb模型文件的节点名称](../../common_operation/read_pb_model_node_name.md)获取。

## 样例执行

以MobileNetV2模型为例，执行在线推理样例。

1. 将.pb模型转换为**SavedModel模型**。
    1. 单击[链接](https://gitee.com/link?target=https%3A%2F%2Fobs-9be7.obs.cn-east-2.myhuaweicloud.com%2F003_Atc_Models%2Fmodelzoo%2FOfficial%2Fcv%2FMobileNetv2_for_ACL.zip)获取MobileNetv2_for_ACL.zip包，将解压后的.pb格式模型（mobileNetv2.pb）移至服务器“$HOME/tf_serving_test“路径。
    2. 在tf_serving_test路径下创建转换脚本pb_to_savedmodel.py，样例代码如下所示：

        ```python
        import tensorflow as tf
        from tensorflow.python.saved_model import signature_constants
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.framework import convert_to_constants
        from tensorflow.python.framework import tensor_shape
        from tensorflow.python.saved_model import save
        import sys

        def read_pb_model(pb_model_path):
            with tf.gfile.GFile(pb_model_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                return graph_def
        def convert_pb_saved_model(graph_def, export_dir, input_name, output_name):
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

            sigs = {}
            with tf.Session(graph=tf.Graph()) as sess:
                tf.import_graph_def(graph_def, name="")
                g = tf.get_default_graph()
                input_name_list = input_name.strip().split(";")
                output_name_list = output_name.strip().split(';')
                input_dict = {}
                output_dict = {}
                for i, s in enumerate(input_name_list):
                    ss = s.split(':')[0] + ' : ' + s.split(':')[0] + ','
                    print(ss)
                    d = g.get_tensor_by_name(s)
                    input_dict.update({s:d})
                for i, s in enumerate(output_name_list):
                    d = g.get_tensor_by_name(s)
                    output_dict.update({s:d})
                out = g.get_tensor_by_name(output_name)
                sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                    tf.saved_model.signature_def_utils.predict_signature_def(
                        input_dict, output_dict)
                builder.add_meta_graph_and_variables(sess,
                                                    [tag_constants.SERVING],
                                                    signature_def_map=sigs)
                builder.save()
        def convert_pb_to_server_model(pb_model_path, export_dir, input_name='input', output_name='output'):
            graph_def = read_pb_model(pb_model_path)
            convert_pb_saved_model(graph_def, export_dir, input_name, output_name)


        # 转换pb文件为savedmodel
        if __name__=="__main__":
            pb_model_path = sys.argv[1]
            export_dir = sys.argv[2]
            input_name = "input:0" # 原始.pb模型输入节点名，请根据实际名称自行修改。如果是多输入节点时，节点名之间使用;隔开。
            output_name = "MobilenetV2/Logits/output:0" # 原始.pb模型输出节点名，请根据实际名称自行修改。如果是多输出节点时，节点名之间使用;隔开。
            convert_pb_to_server_model(pb_model_path, export_dir, input_name, output_name)
        ```

        > [!NOTE]说明
        > 用户在不确定原始.pb模型输入输出节点名时，可参考[读取pb模型文件的节点名称](../../common_operation/read_pb_model_node_name.md)。

    3. 执行以下命令进行转换。

        ```bash
        python3 pb_to_savedmodel.py mobileNetv2.pb ./mobileNetv2
        ```

        参数解释：

        pb_to_savedmodel.py：转换脚本名称。

        mobileNetv2.pb：待转换原始.pb模型名称。

        mobileNetv2：saved_model.pb模型文件输出文件路径。

    4. 将生成的saved_model.pb模型按照如下目录结构存放。

        ```text
        tf_serving_test/
         └── mobileNetv2
            └── 1
               ├── saved_model.pb
               └── variables
        ```

    5. 准备数据集。

        在安装用户$HOME目录下，创建data文件夹，用于存放数据集。

        > [!NOTE]说明
        > 此样例以一张格式为.jpg图片数据进行推理，如需使用其他数据集，请用户自行准备。

2. 在任意目录下执行以下命令启动tensorflow_model_server，启动成功如下图所示。

    ```bash
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_base_path=$HOME/tf_serving_test/mobileNetv2 --model_name=mobileNetv2 --platform_config_file=$HOME/tf_serving_test/config.cfg
    ```

    ![](../../figures/start_success.png "启动成功")

3. 使用安装用户重新登录服务器，进入“Client“目录编辑与服务端通信脚本和推理脚本。

    创建“deploy.py“通信脚本和“tf_serving_infer.py“推理脚本，并参考[样例代码](#样例代码)写入相关代码。

4. 执行如下命令执行推理。

    ```bash
    python3 tf_serving_infer.py $HOME/data/cat.jpg
    ```

5. 推理结果如下图所示。

    ![](../../figures/infer_result.png)
