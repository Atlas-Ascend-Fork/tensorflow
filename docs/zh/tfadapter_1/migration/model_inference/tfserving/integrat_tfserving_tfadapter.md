# TF Serving集成TF Adapter

本节介绍TF Serving集成TF Adapter的操作方法，实际操作时请根据实际路径进行替换。本文中举例路径均需要确保安装用户具有读写权限。

1. 下载[TF Serving源码](https://github.com/tensorflow/serving/archive/1.15.0.zip)。

    TF Serving需与TensorFlow保持版本一致，将源码包上传至服务器任意路径下，本节以$HOME目录为例。

2. 进入源码包所在路径，执行如下命令解压并进入TF Serving源码包。

    ```bash
    unzip 1.15.0.zip
    cd serving-1.15.0/
    ```

3. 添加TF Serving第三方依赖。
    1. 执行如下命令，在“serving-1.15.0/third_party“目录下创建“tf_adapter“文件夹并进入。

        ```bash
        cd third_party/
        mkdir tf_adapter
        cd tf_adapter
        ```

    2. 执行如下命令，在“tf_adapter“文件夹下拷贝存放“libpython3.7m.so.1.0“文件，并创建软链接。

        ```bash
        cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 .
        ln -s libpython3.7m.so.1.0 libpython3.7m.so
        ```

    3. 执行如下命令，在“tf_adapter“文件夹下拷贝存放“_tf_adapter.so“文件，并将“_tf_adapter.so“文件名修改为“lib_tf_adapter.so“。

        ```bash
        cp ${TFPLUGIN_INSTALL_PATH}/npu_bridge/_tf_adapter.so .
        mv _tf_adapter.so lib_tf_adapter.so
        ```

        其中$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包安装路径。

4. 编译生成libtensorflow_framework.so、_pywrap_tensorflow_internal.so文件。
    1. 在“tf_adapter”文件夹下，执行如下命令。

        ```bash
        vim CMakeLists.txt
        ```

    2. 写入如下内容并保存。

        ```text
        file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/stub.c)
        add_library(_pywrap_tensorflow_internal SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub.c)
        add_library(tensorflow_framework SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub.c)
        ```

    3. 执行**:wq!**命令保存文件并退出。
    4. 执行如下命令，编译生成.so文件。

        ```bash
        mkdir temp
        cd temp
        cmake ..
        make
        mv lib_pywrap_tensorflow_internal.so ../_pywrap_tensorflow_internal.so
        mv libtensorflow_framework.so ../libtensorflow_framework.so
        cd ..
        ln -s libtensorflow_framework.so libtensorflow_framework.so.1
        ```

    5. 配置环境命令。

        ```bash
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
        ```

5. 创建BUILD文件并添加内容。
    1. 执行以下命令，在“tf_adapter“文件夹下创建BUILD文件。

        ```bash
        vim BUILD
        ```

    2. 写入如下内容。

        ```text
        licenses(["notice"])  # BSD/MIT.

        cc_import(
            name = "tf_adapter",
            shared_library = "lib_tf_adapter.so",
            visibility = ["//visibility:public"]
        )

        cc_import(
            name = "tf_python",
            shared_library = "libpython3.7m.so",
            visibility = ["//visibility:public"]
        )
        ```

    3. 执行**:wq!**命令保存文件并退出。

6. 修改“serving-1.15.0/tensorflow_serving/model_servers/“路径下的BUILD文件，在“cc_binary“的deps中添加如下所示的后三行代码。

    ```text
    cc_binary(
        name = "tensorflow_model_server",
        stamp = 1,
        visibility = [
            ":testing",
            "//tensorflow_serving:internal",
        ],
        deps = [
            ":tensorflow_model_server_main_lib",
            "//third_party/tf_adapter:tf_adapter",
            "//third_party/tf_adapter:tf_python",
            "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
        ],
    )
    ```

7. 编译TF Serving。

    在TF Serving安装目录“serving-1.15.0“下执行如下命令，编译TF Serving。

    ```bash
    bazel --output_user_root=/opt/tf_serving build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" tensorflow_serving/model_servers:tensorflow_model_server
    ```

    其中“–output_user_root“参数指定了TF Serving的安装路径。请根据实际进行指定。

    > [!NOTE]说明
    >
    > - 如果编译过程中遇到依赖包下载失败问题，可手动下载，参考[手动下载TF Serving编译依赖包](common_operation.md#手动下载tf-serving编译依赖包)。
    > - 如果在TF Serving编译过程中出现“builtins“依赖模块查询失败问题，参考[TF Serving编译时提示缺少builtins](FAQ.md#tf-serving编译时提示缺少builtins)解决。

8. 建立软链接。

    执行如下命令，创建TF Serving软件包的软链接。

    ```bash
    ln -s /opt/tf_serving/{tf_serving_ID}/execroot/tf_serving/bazel-out/xxx-opt/bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/tensorflow_model_server
    ```

    - _\{tf_serving_ID\}_为一串如“063944eceea3e72745362a0b6eb12a3c“的无规则字符。请根据实际进行填写。
    - _xxx-opt_为工具自动生成文件夹，具体显示请以实际为准。
