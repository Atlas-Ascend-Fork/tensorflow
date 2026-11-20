# 如何基于TensorFlow的C++接口适配NPU设备

## 场景说明

开发者将TensorFlow源码与业务代码同时编译使用的场景下，文档中提供的基于Python将TensorFlow脚本迁移到NPU执行训练的方法不再适用。此处为用户提供了一种在TensorFlow源码层面适配NPU设备的方法（此方法仅供参考）。

## 操作步骤

1. 编译代码时，需要在链接动态库选项中增加一个到_tf_adapter.so的链接，这个so是一个让TensorFlow能找到NPU设备的插件。

    以gcc为例，假设“/usr/local/lib/python3.7/site-packages/npu_bridge/_tf_adapter.so”为此so的存储路径，使用方法如下：

    ![](../figures/tf_cpp_calls.png)

2. 对TensorFlow的SessionOptions做适配，使其能将图下发到NPU上计算。

    TensorFlow的SessionOptions变量可以通过以下代码获取：

    ```python
    tensorflow::SessionOptions sessOpts = tensorflow::SessionOptions();
    ```

    接着对此SessionOptions的内容增加以下内容：

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

3. 设置NPU基础环境变量，使其能找到NPU的相关so。

    ```bash
    # 以root用户默认安装路径为例
    source /usr/local/Ascend/cann/set_env.sh
    ```

    完成上述步骤后，即可以在NPU上执行业务了。
