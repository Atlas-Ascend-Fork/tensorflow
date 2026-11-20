# 调用TensorFlow的C++接口在NPU设备执行方法

## 场景说明

开发者将TensorFlow源码与业务代码同时编译使用的场景下，文档中提供的基于Python将TensorFlow脚本迁移到NPU执行训练的方法则不再适用。此处为用户提供了一种在TensorFlow源码层面适配使用NPU设备的方法（此方法仅供参考）。

## 操作步骤

1. 在tf2.x的设备上，编译代码时，需要在链接动态库选项中增加一个到_tf_adapter.so和_npu_ops.so的链接，这个so是一个让TensorFlow能找到NPU设备的插件。

    以gcc为例，“/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/_tf_adapter.so”和"/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/_npu_ops.so"为此so的存储路径，使用示例如下：

    ```bash
    gcc pixellink.cpp -I /home/test/env/lib/python3.7/site-packages/tensorflow/include/ -Wl,--no-as-needed /usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/_tf_adapter.so /usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/_npu_ops.so -Wl,--no-as-needed /usr/local/Ascend/latest/x86_64-linux/lib64/libindextransform.so /home/test/env/lib/python3.7/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so /home/test/env/lib/python3.7/site-packages/tensorflow/libtensorflow_framework.so.2 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 -Wl,-rpath,/home/test/env/lib/python3.7/site-packages/tensorflow/python/:/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/:/home/test/env/lib/python3.7/site-packages/tensorflow/:/usr/local/Ascend/latest/compiler/lib64/:/usr/local/Ascend/latest/runtime/lib64/:/usr/local/Ascend/driver/lib64/driver/:/lib/x86_64-linux-gnu/:/usr/local/python3.7.5/lib/:/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/compat/v1/:/usr/local/python3.7.5/lib/python3.7/site-packages/npu_device/ -o demo_tf -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0
    ```

2. 对TensorFlow的SessionOptions做适配，使其能将图下发到NPU上计算。

    TensorFlow的SessionOptions变量可以通过以下代码获取：

    ```python
    tensorflow::SessionOptions sessOpts = tensorflow::SessionOptions();
    ```

    修改SessionOptions，增加如下内容：

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

    后续，就可以在NPU上执行业务了。
