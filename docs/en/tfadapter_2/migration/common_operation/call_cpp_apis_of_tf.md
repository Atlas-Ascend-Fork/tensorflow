# Calling C++ APIs of TensorFlow to Execute Methods on NPUs

## Scenario Description

In the scenario where developers compile TensorFlow source code and service code at the same time, the method provided in the document for porting TensorFlow scripts to NPUs for training using Python is no longer applicable. This section provides a method for adapting NPUs at the TensorFlow source code level \(for reference only\).

## Procedure

1. When compiling code on the TF 2.x device, add a link to **_tf_adapter.so** and **_npu_ops.so** in the dynamic library link options. These .so files are a plugin that enables TensorFlow to find the NPU device.

    Take gcc as an example. **/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/_tf_adapter.so** and **/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/_npu_ops.so** are the storage paths of the .so files. The following shows the details.

    ```bash
    gcc pixellink.cpp -I /home/test/env/lib/python3.7/site-packages/tensorflow/include/ -Wl,--no-as-needed /usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/_tf_adapter.so /usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/_npu_ops.so -Wl,--no-as-needed /usr/local/Ascend/latest/x86_64-linux/lib64/libindextransform.so /home/test/env/lib/python3.7/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so /home/test/env/lib/python3.7/site-packages/tensorflow/libtensorflow_framework.so.2 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 -Wl,-rpath,/home/test/env/lib/python3.7/site-packages/tensorflow/python/:/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/:/home/test/env/lib/python3.7/site-packages/tensorflow/:/usr/local/Ascend/latest/compiler/lib64/:/usr/local/Ascend/latest/runtime/lib64/:/usr/local/Ascend/driver/lib64/driver/:/lib/x86_64-linux-gnu/:/usr/local/python3.7.5/lib/:/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/:/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/ -o demo_tf -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0
    ```

2. Adapt **SessionOptions** of TensorFlow so that graphs can be delivered to the NPU for computation.

    You can obtain the **SessionOptions** variable of TensorFlow by using the following code:

    ```python
    tensorflow::SessionOptions sessOpts = tensorflow::SessionOptions();
    ```

    Add the following content to **SessionOptions**:

    ```python
    auto *custom_op =
    sessOpts.config.mutable_graph_options()->mutable_rewrite_options()->add_custom_optimizers();
    custom_op->set_name("NpuOptimizer");
    AttrValue value_bool;
    value_bool.set_b(true);
    custom_op->mutable_parameter_map()->insert({"use_off_line", value_bool});
    AttrValue value_string;
    value_string.set_s(std::string("force_fp16"));
    custom_op->mutable_parameter_map()->insert({"precision_mode", value_string});
    AttrValue value_int;
    value_int.set_i(0);
    custom_op->mutable_parameter_map()->insert({"graph_run_mode", value_int});
    sessOpts.config.mutable_graph_options()->mutable_rewrite_options()->set_remapping(RewriterConfig::OFF);
    sessOpts.config.mutable_graph_options()->mutable_rewrite_options()->set_memory_optimization(RewriterConfig_MemOptType_MANUAL);
    ```

3. Set the basic NPU environment variable so that it can find the .so file related to the NPU.

    ```bash
    # The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh
    ```

    Then, services can be executed on the NPU.
