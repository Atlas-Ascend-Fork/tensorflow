# How Do I Adapt TensorFlow to NPU Devices Using C++ APIs?

## Scenario Description

In the scenario where developers compile TensorFlow source code and service code at the same time, the method provided in the document for porting TensorFlow scripts to NPUs for training using Python is no longer applicable. This section provides a method for adapting to NPU devices at the TensorFlow source code level \(for reference only\).

## Procedure

1. When compiling code, add a link to **_tf_adapter.so** to the dynamic library link option. This .so file is a plugin that enables TensorFlow to find a specified NPU device.

    Take GCC as an example. Assume that **/usr/local/lib/python3.7/site-packages/npu_bridge/_tf_adapter.so** is the storage path of the .so file. The usage method is as follows.

    ![](../figures/tf_cpp_calls.png)

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

    After the preceding steps are complete, services can be executed on the NPU.
