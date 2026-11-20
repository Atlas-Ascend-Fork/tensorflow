# Common Operations

## Installing bazel 0.24.1 from Source Code

1. Install the system dependencies. Ubuntu and CentOS are used as examples.

    - Ubuntu 18.04 x86_64:

        ```bash
        apt-get install build-essential openjdk-11-jdk python zip unzip
        ```

    - CentOS 8.3 AArch64:

        ```bash
        yum install java-11-openjdk-devel.aarch64
        yum install java-11-openjdk.aarch64
        yum groupinstall 'Development Tools'
        yum install zip
        ```

    If java-11-openjdk fails to be installed, manually install it by referring to  [Installing java-11-openjdk1 Manually](#installing-java-11-openjdk1-manually).

2. Set environment variables.
    1. Open the **.bashrc** file.

        ```bash
        vim ~/.bashrc
        ```

    2. Add the java-11-openjdk installation path to the file. \(The path in the following command is simply an example. Replace it with the actual installation path.\)

        ```bash
        export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
        export PATH=$JAVA_HOME/bin:$PATH
        ```

    3. Run the **:wq!** command to save the file and exit.
    4. Run the following command for the environment variables to take effect:

        ```bash
        source ~/.bashrc
        ```

3. Download the  [bazel compressed source package](https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip)  and upload it to any directory on the server.
4. Go to the directory where the source package is stored and perform compilation and installation.
    1. Decompress the downloaded bazel source package.

        ```bash
        unzip bazel-0.24.1-dist.zip -d bazel-0.24.1-dist
        ```

    2. Go to the decompressed folder, and run the following configuration, compilation, and installation commands:

        ```bash
        cd bazel-0.24.1-dist/
        env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" ./compile.sh
        cp output/bazel /usr/local/bin
        ```

5. Validate your installation.

    After the installation is complete, run the following command to check the version:

    ```bash
    bazel --version
    ```

## Installing CMake 3.14.0 from Source Code

1. Download the  [CMake compressed source package](https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz)  and upload it to any directory on the server.
2. Go to the directory where the source package is stored and perform compilation and installation.
    1. Decompress the downloaded CMake source package.

        ```bash
        tar -zxvf cmake-3.14.0.tar.gz
        ```

    2. Go to the decompressed folder, and run the following configuration, compilation, and installation commands:

        ```bash
        cd cmake-3.14.0/
        ./bootstrap --prefix=/usr
        make -j4
        make install
        ```

3. Validate your installation.

    After the installation is complete, run the following command to check the version:

    ```bash
    cmake --version
    ```

## Installing java-11-openjdk1 Manually

Three packages are required for installing **java-11-openjdk**. You can go to [https://centos.pkgs.org](https://centos.pkgs.org/)  and search for **java-11-openjdk** in the upper-right corner to obtain URLs of the software packages with the **.rpm** extension.  The following figure is an example of the search result. Download the corresponding software based on the system version, and then run the **rpm** command to install the software.

![](../../figures/network_search_result.png "search-result")

The following uses CentOS 8.3 AArch64 as an example to describe how to install **java-11-openjdk** software packages.

| Software Package | Package Download Path |
| --- | --- |
| java-11-openjdk-headless | [Link](https://vault.centos.org/centos/8/AppStream/aarch64/os/Packages/java-11-openjdk-headless-11.0.13.0.8-4.el8_5.aarch64.rpm) |
| java-11-openjdk | [Link](https://vault.centos.org/centos/8/AppStream/aarch64/os/Packages/java-11-openjdk-11.0.13.0.8-4.el8_5.aarch64.rpm) |
| java-11-openjdk-devel | [Link](https://vault.centos.org/centos/8/AppStream/aarch64/os/Packages/java-11-openjdk-devel-11.0.13.0.8-4.el8_5.aarch64.rpm) |

1. Download the  **.rpm**  package listed in the preceding table and upload the software package to any directory on the server.
2. Go to the directory where the software package is stored and install the  **.rpm**  package.

    ```bash
    rpm -ivh java-11-openjdk-headless-11.0.13.0.8-4.el8_5.aarch64.rpm
    rpm -ivh java-11-openjdk-11.0.13.0.8-4.el8_5.aarch64.rpm
    rpm -ivh java-11-openjdk-devel-11.0.13.0.8-4.el8_5.aarch64.rpm
    ```

## Manually Downloading Dependency Packages for TF Serving Build

Dependency packages need to be downloaded from the network during TF Serving build. If the download fails due to network problems, the error message shown in the following figure is displayed.

![tf-serving-compilation-error](../../figures/TF-Serving_compile_error.png)

The following describes the solution.

1. Download the required dependency packages via the following links and upload them to any directory on the server \(for example, **$\{HOME\}**\):

    [tensorflow](https://github.com/tensorflow/tensorflow/archive/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b.tar.gz),  [rules_closure](https://github.com/bazelbuild/rules_closure/archive/316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz),  [bazel-skylib](https://github.com/bazelbuild/bazel-skylib/archive/0.7.0.tar.gz),  [rapidjson](https://github.com/Tencent/rapidjson/archive/v1.1.0.zip),  [abseil-cpp](https://github.com/abseil/abseil-cpp/archive/36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz),  [libevent](https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip), and  [llvm](https://github.com/llvm-mirror/llvm/archive/7a7e03f906aada0cf4b749b51213fe5784eeff84.tar.gz)

    The downloaded dependency packages can be used only after being renamed, as shown in the following table.

    | Dependency | Before | After |
    | --- | --- | --- |
    | tensorflow | tensorflow-590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b.tar.gz | 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b.tar.gz |
    | rules_closure | rules_closure-316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz | 316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz |
    | bazel-skylib | bazel-skylib-0.7.0.tar.gz | 0.7.0.tar.gz |
    | rapidjson | rapidjson-1.1.0.zip | v1.1.0.zip |
    | abseil-cpp | abseil-cpp-36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz | 36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz |
    | libevent | libevent-release-2.1.8-stable.zip | release-2.1.8-stable.zip |
    | llvm | llvm-7a7e03f906aada0cf4b749b51213fe5784eeff84.tar.gz | 7a7e03f906aada0cf4b749b51213fe5784eeff84.tar.gz |

2. Add the **--distdir** parameter when compiling TF Serving as follows:

    ```bash
    bazel --output_user_root=/opt/tf_serving build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --distdir=${HOME}/tensorflow_serving/model_servers:tensorflow_model_server
    ```

## Converting a SavedModel to an .om Model

This section describes how to use the  [saved_model2om.py](https://gitee.com/ascend/tools/tree/master/saved_model2om)  tool to convert a trained  **SavedModel**  into an .om model. Using the converted .om model during TF Serving deployment shortens the compilation time and improves the TF Serving deployment performance.

> [!CAUTION]NOTICE
> When an .om model is used for online inference, the data dump function for accuracy comparison is not supported.

### Options

| Option | Description | Example Value |
| --- | --- | --- |
| --input_path | - Input path of the original SavedModel file.<br>  - Required. | $HOME/inputpath/model |
| --output_path | - Output path of the SavedModel file generated after conversion.<br>  - Required. | $HOME/outputpath/model |
| --input_shape | - Shape value of the input model. The format is name1:shape;name2:shape;name3:shape. When input_shape is set, the dimensions that are not clearly defined in the shape are automatically set to 1.<br>  - Optional. | input:16,224,224,3 |
| --soc_version | - SoC version of the output .om model. You do not need to set it if --profiling is set. The value is determined by the device that performs the conversion.<br>  - Required. | Ascendxxx |
| --profiling | - When this option is set, AOE optimization is enabled. (You do not need to specify job_type.)1: enables subgraph tuning.2: enables operator tuning.<br>  - 1: enables subgraph tuning.<br>  - 2: enables operator tuning.<br>  - Set this option if subgraph or operator tuning is required. | 1 |
| --method_name | - Inference method when the TF Serving runtime is configured. If this parameter is not specified, the inference method is obtained from the original SavedModel file.<br>  - Optional. | /tensorflow/serving/predict |
| --new_input_nodes | - Re-selects an input node. The format is "Operator:Type:Operator name;Operator:Type:Operator name".<br>  - Optional. | embedding:DT_FLOAT:bert/embedding/word_embeddings:0;add:DT_INT:bert/embedding/add:0 |
| --new_output_nodes | - Re-selects an output node. The format is "Operator:Operator name".<br>  - Optional. | loss:loss/Softmax:0 |
| --output_type | - Sets the output data type of the network or an output node. For details about how to use the option, see [AOE Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/aoe/auxiliarydevtool_aoe_0001.html) or [ATC Offline Model Builder](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/atctool/atlasatc_16_0001.html).<br>  - Optional. | node1:0:FP16 |
| --input_fp16_nodes | - Sets the name of the input node whose input data type is FP16. For details about how to use the option, see [AOE Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/aoe/auxiliarydevtool_aoe_0001.html) or [ATC Offline Model Builder](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/atctool/atlasatc_16_0001.html).<br>  - Optional. | node_name1;node_name2 |

> [!NOTE]NOTE
>This tool supports both ATC and AOE parameters.
>
>- If  **--profiling**  is set, see  [AOE Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/aoe/auxiliarydevtool_aoe_0001.html).
>- If  **--profiling**  is not set, see  [ATC Offline Model Builder](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/atctool/atlasatc_16_0001.html).
>This tool does not support the  **--out_nodes**,  **--is_input_adjust_hw_layout**, and  **--is_output_adjust_hw_layout**  options of the ATC and AOE tools. The  **--out_nodes**  option can be replaced by the  **--new_output_nodes**  option in  in the preceding table.

## Conversion

1. Download the conversion tool  [saved_model2om.py](https://gitee.com/ascend/tools/tree/master/saved_model2om)  to any directory on the server, for example,  _$HOME/tools/_. You do not need to install the tool.
2. Run the following command to perform the conversion. Modify the parameters based on the actual situation.

    ```bash
    python3 saved_model2om.py --input_path "$HOME/inputpath/model" --output_path "$HOME/outputpath/model" --input_shape "input:16,224,224,3" --soc_version "Ascendxxx"
    ```

    For the method of querying the value of **--soc_version**, please refer to the  [ATC Offline Model Builder](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/atctool/atlasatc_16_0001.html).

    If subgraph or operator tuning is required during the conversion, run the following command:

    ```bash
    python3 saved_model2om.py --input_path "$HOME/inputpath/model" --output_path "$HOME/outputpath/model" --input_shape "input:16,224,224,3" --profiling "1"
    ```

3. After the conversion is successful, a SavedModel file for loading the OM model is generated in the specified  **output_path**. The file name format is  **_\{om_name\}**load_om_saved_model**\{timestamp\}_**.

## Rebuilding TF Serving

After the CANN software of another version is installed, the  **tensorflow_model_server**  service may fail to be started due to a dynamic link library \(DLL\) link error, as shown in  the following figure.

![](../../figures/dll_linking_error.png)

The following describes the solution.

1. Go to the  **serving-1.15.0/third_party/tf_adapter**  directory and run the following commands.

    Copy and save the  **_tf_adapter.so**  file in the  **tf_adapter**  folder, and change the name of the  **_tf_adapter.so**  file to  **lib_tf_adapter.so**.

    ```bash
    cp ${TFPLUGIN_INSTALL_PATH}/npu_bridge/_tf_adapter.so .
    mv _tf_adapter.so lib_tf_adapter.so
    ```

2. Run the following commands to clear the cache from the previous compilation and avoid incremental compilation:

    ```bash
    rm -rvf /opt/tf_serving
    bazel clean
    ```

3. Build TF Serving.

    Run the following command in the TF Serving installation directory  **serving-1.15.0**  to build TF Serving:

    ```bash
    bazel --output_user_root=/opt/tf_serving build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" tensorflow_serving/model_servers:tensorflow_model_server
    ```

    The  **--output_user_root**  option specifies the installation path of TF Serving. Set it based on the actual situation.
