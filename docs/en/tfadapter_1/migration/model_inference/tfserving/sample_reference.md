# Sample Reference

## Sample Code

Create a  **Client**  folder in any directory \(for example,  **$HOME**\) to store the  **deploy.py**  script for communication with the server and the  **tf_serving_infer.py**  inference script. The directory is as follows:

```text
Client/
 ├── deploy.py
 └── tf_serving_infer.py
```

Sample code  **deploy.py**:

```python
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np
import os
import time

class PredictModelGrpc(object):
    def __init__(self, model_name, input_name, output_name, socket='xxx.xxx.xxx.xxx:8500'):# xxx.xxx.xxx.xxx is the IP address of the server.
        self.socket = socket
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.request, self.stub = self.__get_request()
    def __get_request(self):
        channel = grpc.insecure_channel(self.socket, options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                                              ('grpc.max_receive_message_length',
                                                               1024 * 1024 * 1024)])  # The size is configurable.
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()

        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"

        return request, stub
    def inference(self, frames):

        t0 = time.time()
        self.request.inputs[self.input_name].CopyFrom(tf.make_tensor_proto(frames, dtype=tf.float32))# Send a request.
        t1 = time.time()
        result = self.stub.Predict.future(self.request, 1000.0) # Start inference. You are advised to set the request wait time to 1000.0.
        t2 = time.time()

        res = []
        res.append(tf.make_ndarray(result.result().outputs[self.output_name])[0]) # Obtain the result.

        t3 = time.time()
        print("Time cost: request.inputs={:.3f} ms, Predict.future={:.3f} ms, get output={:.3f} ms".format((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000))# Print the consumed time.
        return res
```

Sample code  **tf_serving_infer.py**:

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
# Data preprocessing
d0 = time.time()
image = misc.imread(image_path)
resized = scipy.misc.imresize(image, (304, 304, 3)) # Type value of the input node of the original .pb model. Set it based on the actual situation.
crop_min = abs(304 / 2 - (304 / 2))
crop_max = crop_min + 304
crop_min = int(crop_min)
crop_max = int(crop_max)
image = resized[crop_min:crop_max, crop_min:crop_max, :]

mean_sub = image.astype(np.float32) - np.array([123, 117, 104]).astype(np.float32) # Obtain the data type from the type value of the input node of the original .pb model. Set it based on the actual situation.
image = np.expand_dims(np.array(mean_sub), 0)

d1 = time.time()
data_process_time.append(d1 - d0)
model = PredictModelGrpc(model_name='mobileNetv2', input_name='input:0', output_name='MobilenetV2/Logits/output:0') # Set it based on the actual model name, model input node name, and model output node name. If there are multiple input and output nodes, separate them with semicolons (;).

# Start inference.
infer_cost_time = []
for i in range(1000):
    t0 = time.time()
    res = model.inference(image)
    t1 = time.time()
    infer_cost_time.append(t1 - t0)
    print("Index= {}, Inference time cost={:.3f} ms".format(i,(t1-t0)*1000))

# Print the inference time.
print("Batchsize={}, Average inference time cost: {:.3f} ms, Average data process time cost:{:.3f} ms".format(1, (sum(infer_cost_time) - infer_cost_time[0]) / (len(infer_cost_time)) * 1000, (sum(data_process_time) - data_process_time[0]) / (len(data_process_time)) * 1000))
```

> [!NOTE]NOTE
>If you are not sure about the input and output node names and type values of the original .pb model, refer to  [Reading Node Names from a PB Model File](../../common_operation/read_pb_model_node_name.md)  to obtain them.

## Sample Execution

The following uses the MobileNetV2 model as an example to describe how to perform online inference.

1. Convert the  **.pb**  model to the  **SavedModel model**.
    1. Click  [here](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/MobileNetv2_for_ACL.zip)  to obtain the  **MobileNetv2_for_ACL.zip**  package and save the decompressed .pb model \(**mobileNetv2.pb**\) to the  **$HOME/tf_serving_test**  directory on the server.
    2. Create the conversion script  **pb_to_savedmodel.py**  in  **tf_serving_test**. The sample code is as follows:

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


        # Convert the .pb file to SavedModel.
        if __name__=="__main__":
            pb_model_path = sys.argv[1]
            export_dir = sys.argv[2]
            input_name = "input:0" # Input node name of the original .pb model. Set it based on the actual situation. If there are multiple input nodes, separate the node names with semicolons (;).
            output_name = "MobilenetV2/Logits/output:0" # Output node name of the original .pb model. Set it based on the actual situation. If there are multiple output nodes, separate the node names with semicolons (;).
            convert_pb_to_server_model(pb_model_path, export_dir, input_name, output_name)
        ```

        > [!NOTE]NOTE
        >If you are not sure about the input and output node names of the original  **.pb**  model, see  [Reading Node Names from a PB Model File](../../common_operation/read_pb_model_node_name.md).

    3. Run the following command to start conversion.

        ```bash
        python3 pb_to_savedmodel.py mobileNetv2.pb ./mobileNetv2
        ```

        The parameters are described as follows:

        **pb_to_savedmodel.py**: conversion script name.

        **mobileNetv2.pb**: name of the original  **.pb**  model to be converted.

        **mobileNetv2**: output file path of the  **saved_model.pb**  model file.

    4. Save the generated  **saved_model.pb**  model based on the following directory structure:

        ```text
        tf_serving_test/
         └── mobileNetv2
            └── 1
               ├── saved_model.pb
               └── variables
        ```

    5. Prepare a dataset.

        Create a  **data**  folder in the  **$HOME**  directory of the installation user to store the dataset.

        > [!NOTE]NOTE
        > In this sample, a  **.jpg**  image is used for inference. You can also customize your dataset.

2. Run the following command in any directory to start  **tensorflow_model_server**. If information shown in the following figure  is displayed,  **tensorflow_model_server**  is successfully started.

    ```bash
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_base_path=$HOME/tf_serving_test/mobileNetv2 --model_name=mobileNetv2 --platform_config_file=$HOME/tf_serving_test/config.cfg
    ```

    ![](../../figures/start_success.png "successful-startup")

3. Log in to the server again as the installation user, go to the  **Client**  directory, and edit the script for communicating with the server and the script for inference.

    Create the  **deploy.py**  communication script and  **tf_serving_infer.py**  inference script, and refer to  [Sample Code](#sample-code)  to write related code.

4. Run the following command to start inference:

    ```bash
    python3 tf_serving_infer.py $HOME/data/cat.jpg
    ```

5. See the following figure for the inference result.

    ![](../../figures/infer_result.png)
