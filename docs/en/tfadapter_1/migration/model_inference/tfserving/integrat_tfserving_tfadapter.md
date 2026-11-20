# Integrating TF Serving with TF Adapter

This section describes how to integrate TF Serving with TF Adapter. Replace the example paths with your actual paths as needed. Ensure that the installation user has the read and write permissions on the paths described in this document.

1. Download the  [TF Serving source code](https://github.com/tensorflow/serving/archive/1.15.0.zip).

    The version of TF Serving must be the same as that of TensorFlow. Upload the source package to any directory on the server. This section uses the  **$HOME**  directory as an example.

2. Go to the directory where the source package is stored and run the following commands to decompress and access the TF Serving source package:

    ```bash
    unzip 1.15.0.zip
    cd serving-1.15.0/
    ```

3. Add third-party dependency packages of TF Serving.
    1. Create the  **tf_adapter**  folder in the  **serving-1.15.0/third_party**  directory, and go to the folder.

        ```bash
        cd third_party/
        mkdir tf_adapter
        cd tf_adapter
        ```

    2. Copy the  **libpython3.7m.so.1.0**  file to the  **tf_adapter**  folder, and create a soft link.

        ```bash
        cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 .
        ln -s libpython3.7m.so.1.0 libpython3.7m.so
        ```

    3. Copy the  **_tf_adapter.so**  file to the  **tf_adapter**  folder, and change the file name from  **_tf_adapter.so**  to  **lib_tf_adapter.so**.

        ```bash
        cp ${TFPLUGIN_INSTALL_PATH}/npu_bridge/_tf_adapter.so .
        mv _tf_adapter.so lib_tf_adapter.so
        ```

        $\{TFPLUGIN_INSTALL_PATH\}  is the installation path of the TF Adapter package.

4. Compile and generate the  **libtensorflow_framework.so**  and  **_pywrap_tensorflow_internal.so**  files.
    1. Run the following command in the  **tf_adapter**  folder:

        ```bash
        vim CMakeLists.txt
        ```

    2. Add the following content, and save the file.

        ```text
        file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/stub.c)
        add_library(_pywrap_tensorflow_internal SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub.c)
        add_library(tensorflow_framework SHARED ${CMAKE_CURRENT_BINARY_DIR}/stub.c)
        ```

    3. Run the  **:wq!**  command to save the file and exit.
    4. Compile and generate the .so file.

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

    5. Configure the environment variable.

        ```bash
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
        ```

5. Create the BUILD file, and add the following content:
    1. Run the following commands to create a BUILD file in the  **tf_adapter**  folder:

        ```bash
        vim BUILD
        ```

    2. Add the following content:

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

    3. Run the  **:wq!**  command to save the file and exit.

6. Add the following last three lines of code to deps of  **cc_binary**  in the BUILD file under the  **serving-1.15.0/tensorflow_serving/model_servers/**  directory:

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

7. Perform TF Serving build.

    Run the following command in the TF Serving installation directory  **serving-1.15.0**  to build TF Serving:

    ```bash
    bazel --output_user_root=/opt/tf_serving build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" tensorflow_serving/model_servers:tensorflow_model_server
    ```

    The  **–output_user_root**  option specifies the installation path of TF Serving. Set this parameter based on the actual situation.

    > [!NOTE]NOTE
    >
    >- If downloading of a dependency package fails, manually download it by referring to  [Manually Downloading Dependency Packages for TF Serving Build](common_operation.md#manually-downloading-dependency-packages-for-tf-serving-build).
    >- If querying of  **builtins**  dependency modules fails, rectify the fault by referring to  [What Should I Do If an Error About builtins Missing Is Displayed During TF Serving Build?](FAQ.md#what-should-i-do-if-an-error-about-builtins-missing-is-displayed-during-tf-serving-build).

8. Create a soft link.

    The command used for creating a soft link is as follows:

    ```bash
    ln -s /opt/tf_serving/{tf_serving_ID}/execroot/tf_serving/bazel-out/xxx-opt/bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/tensorflow_model_server
    ```

    - **_\{tf_serving_ID\}_**  is a string of irregular characters, for example,  **063944eceea3e72745362a0b6eb12a3c**. Set this parameter based on the actual situation.
    - The  **_xxx-opt_**  folder is automatically generated by the tool. Replace it with the actual folder name.
