# 常用操作

## 源码安装0.24.1版本bazel

1. 安装系统依赖，此处以Ubuntu与CentOS操作系统为例。

    - Ubuntu 18.04 x86_64环境：

        ```bash
        apt-get install build-essential openjdk-11-jdk python zip unzip
        ```

    - CentOS 8.3 aarch64环境：

        ```bash
        yum install java-11-openjdk-devel.aarch64
        yum install java-11-openjdk.aarch64
        yum groupinstall 'Development Tools'
        yum install zip
        ```

    若java-11-openjdk安装失败，可进行手动安装，参考[手动安装java-11-openjdk](#手动安装java-11-openjdk)。

2. 配置环境变量。
    1. 执行如下命令，打开“.bashrc”文件。

        ```bash
        vim ~/.bashrc
        ```

    2. 在文件中添加java-11-openjdk的安装路径（以下为示例路径，用户需根据实际路径进行设置）。

        ```bash
        export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
        export PATH=$JAVA_HOME/bin:$PATH
        ```

    3. 执行**:wq!**命令保存文件并退出。
    4. 执行如下命令使环境变量生效。

        ```bash
        source ~/.bashrc
        ```

3. 下载[bazel源码压缩包](https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip)，将源码包上传至服务器任意路径下。
4. 进入源码包所在路径，进行编译安装。
    1. 执行如下命令，解压下载的bazel源码压缩包。

        ```bash
        unzip bazel-0.24.1-dist.zip -d bazel-0.24.1-dist
        ```

    2. 进入解压后的文件夹，执行配置、编译和安装命令。

        ```bash
        cd bazel-0.24.1-dist/
        env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" ./compile.sh
        cp output/bazel /usr/local/bin
        ```

5. 安装验证。

    安装完成后重新执行如下命令查看版本号。

    ```bash
    bazel --version
    ```

## 源码安装3.14.0版本CMake

1. 下载[CMake源码压缩包](https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz)，将源码包上传至服务器任意路径下。
2. 进入源码包所在路径，进行编译安装。
    1. 执行如下命令，解压下载的CMake源码压缩包。

        ```bash
        tar -zxvf cmake-3.14.0.tar.gz
        ```

    2. 进入解压后的文件夹，执行配置、编译和安装命令。

        ```bash
        cd cmake-3.14.0/
        ./bootstrap --prefix=/usr
        make -j4
        make install
        ```

3. 安装验证。

    安装完成后重新执行如下命令查看版本号。

    ```bash
    cmake --version
    ```

## 手动安装java-11-openjdk

安装“java-11-openjdk”总共需要安装三个包，用户可通过在[https://centos.pkgs.org](https://centos.pkgs.org/)网站右上角搜索“java-11-openjdk“查询以“.rpm“为后缀的软件包的URL。网站搜索结果如下图所示。根据系统版本选择相应软件进行下载，再通过“rpm“命令安装。

![](../../figures/network_search_result.png "网站搜索截图")

以CentOS 8.3 aarch64环境为例，进行“java-11-openjdk“相关软件包安装演示。

| 软件包名 | 软件包下载路径 |
|----------|----------------|
| java-11-openjdk-headless | [链接](https://vault.centos.org/centos/8/AppStream/aarch64/os/Packages/java-11-openjdk-headless-11.0.13.0.8-4.el8_5.aarch64.rpm) |
| java-11-openjdk | [链接](https://vault.centos.org/centos/8/AppStream/aarch64/os/Packages/java-11-openjdk-11.0.13.0.8-4.el8_5.aarch64.rpm) |
| java-11-openjdk-devel | [链接](https://vault.centos.org/centos/8/AppStream/aarch64/os/Packages/java-11-openjdk-devel-11.0.13.0.8-4.el8_5.aarch64.rpm) |

1. 参考上表下载“.rpm“包，将软件包上传至服务器任意路径下。
2. 进入软件包所在路径，安装“.rpm“包。

    ```bash
    rpm -ivh java-11-openjdk-headless-11.0.13.0.8-4.el8_5.aarch64.rpm
    rpm -ivh java-11-openjdk-11.0.13.0.8-4.el8_5.aarch64.rpm
    rpm -ivh java-11-openjdk-devel-11.0.13.0.8-4.el8_5.aarch64.rpm
    ```

## 手动下载TF Serving编译依赖包

TF Serving编译过程中，需要下载依赖包，可能会因为网络问题下载失败，报错如下图所示：

![TF-Serving编译报错](../../figures/TF-Serving_compile_error.png)

解决方法如下所示：

1. 参考如下链接下载所需依赖包，将依赖包上传至服务器任意路径下（例如：“\$\{HOME\}“）。

    [tensorflow](https://github.com/tensorflow/tensorflow/archive/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b.tar.gz)、[rules_closure](https://github.com/bazelbuild/rules_closure/archive/316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz)、[bazel-skylib](https://github.com/bazelbuild/bazel-skylib/archive/0.7.0.tar.gz)、[rapidjson](https://github.com/Tencent/rapidjson/archive/v1.1.0.zip)、[abseil-cpp](https://github.com/abseil/abseil-cpp/archive/36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz)、[libevent](https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip)和[llvm](https://github.com/llvm-mirror/llvm/archive/7a7e03f906aada0cf4b749b51213fe5784eeff84.tar.gz)。

    下载的依赖包需要重命名才能使用，如下表所示：

    | 依赖包 | 修改前 | 修改后 |
    |--------|--------|--------|
    | tensorflow | tensorflow-590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b.tar.gz | 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b.tar.gz |
    | rules_closure | rules_closure-316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz | 316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz |
    | bazel-skylib | bazel-skylib-0.7.0.tar.gz | 0.7.0.tar.gz |
    | rapidjson | rapidjson-1.1.0.zip | v1.1.0.zip |
    | abseil-cpp | abseil-cpp-36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz | 36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz |
    | libevent | libevent-release-2.1.8-stable.zip | release-2.1.8-stable.zip |
    | llvm | llvm-7a7e03f906aada0cf4b749b51213fe5784eeff84.tar.gz | 7a7e03f906aada0cf4b749b51213fe5784eeff84.tar.gz |

2. 在编译TF Serving时添加“--distdir“参数，如下所示：

    ```bash
    bazel --output_user_root=/opt/tf_serving build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --distdir=${HOME}/tensorflow_serving/model_servers:tensorflow_model_server
    ```

## SavedModel模型转换om模型

开发者可使用[saved_model2om.py](https://gitee.com/ascend/tools/tree/master/saved_model2om)工具将训练保存的SavedModel模型转换为om模型，在部署TF Serving时使用转换后的om模型可以缩短编译时间，从而提升TF Serving部署性能。

> [!NOTE]说明
> 使用.om格式的模型进行在线推理时，不支持精度比对中Data Dump功能。

### 参数说明

| 参数 | 参数说明 | 取值示例 |
|------|----------|----------|
| --input_path | - 原始SavedModel模型文件的输入路径。<br>- 必选。 | $HOME/inputpath/model |
| --output_path | - 转换成功后生成SavedModel模型文件的输出路径。<br>- 必选。 | $HOME/outputpath/model |
| --input_shape | - 输入模型的shape值，格式为"name1:shape;name2:shape;name3:shape"。当设置input_shape时，shape中未明确定义的维度将会被自动设置为1。<br>- 可选。 | input:16,224,224,3 |
| --soc_version | - 输出.om模型的芯片类型。当设置--profiling参数时，无需配置此参数，由当前执行转换的设备决定。<br>- 必选。 | Ascendxxx |
| --profiling | - 设置此参数时，则会开启AOE调优。（该参数配置后无需再指定job_type）。<br>  - 取值为1时，启用子图调优；<br>  - 取值为2时，启用算子调优。<br>- 如需进行子图或者算子调优，则该参数必选。 | 1 |
| --method_name | - 配置TF Serving运行时推理的方法，如果不配置此参数，则会从原始SavedModel模型文件中获取。<br>- 可选。 | /tensorflow/serving/predict |
| --new_input_nodes | - 重新选择输入节点，格式为：算子:类型:算子名;算子:类型:算子名。<br>- 可选。 | embedding:DT_FLOAT:bert/embedding/word_embeddings:0;add:DT_INT:bert/embedding/add:0 |
| --new_output_nodes | - 重新选择输出节点，格式为：算子:算子名。<br>- 可选。 | loss:loss/Softmax:0 |
| --output_type | - 指定网络输出数据类型或指定某个输出节点的输出类型，参数的使用方法请参考《[ATC离线模型编译工具](https://hiascend.com/document/redirect/CannCommunityAtc)》和《[AOE调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolAoe)》。<br>- 可选。 | node1:0:FP16 |
| --input_fp16_nodes | - 指定输入数据类型为FP16的输入节点名称，参数的使用方法请参考《[ATC离线模型编译工具](https://hiascend.com/document/redirect/CannCommunityAtc)》和《[AOE调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolAoe)》。<br>- 可选。 | node_name1;node_name2 |

> [!NOTE]说明
> 该工具同时支持ATC和AOE工具中的参数：
>
> - 设置--profiling参数时请参考《[AOE调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolAoe)》。
> - 未设置--profiling参数时请参考《[ATC离线模型编译工具](https://hiascend.com/document/redirect/CannCommunityAtc)》。
> 该工具暂不支持ATC和AOE工具中的--out_nodes、--is_input_adjust_hw_layout和--is_output_adjust_hw_layout参数，其中--out_nodes参数可使用上表中的--new_output_nodes参数进行代替。

### 执行转换

1. 下载转换工具“[saved_model2om.py](https://gitee.com/ascend/tools/tree/master/saved_model2om)”至服务器的任一目录，例如上传到$HOME/tools/目录下，无需安装。
2. 执行以下命令进行转换，具体参数请根据实际修改。

    ```bash
    python3 saved_model2om.py --input_path "$HOME/inputpath/model" --output_path "$HOME/outputpath/model" --input_shape "input:16,224,224,3" --soc_version "Ascendxxx"
    ```

    --soc_version的取值查询方式请参考《[ATC离线模型编译工具](https://hiascend.com/document/redirect/CannCommunityAtc)》。

    如果在转换的过程中需要进行子图或者算子调优，请执行以下命令。

    ```bash
    python3 saved_model2om.py --input_path "$HOME/inputpath/model" --output_path "$HOME/outputpath/model" --input_shape "input:16,224,224,3" --profiling "1"
    ```

3. 转换成功后，会在指定的output_path下生成用于加载om模型的SavedModel模型文件，文件名格式为\{om_name\}_load_om_saved_model_\{timestamp\}。

## 重新编译TF Serving

重新安装其他版本CANN软件后，直接启动tensorflow_model_server服务，可能会因为动态链接库链接错误导致服务启动失败，报错如下图所示：

![动态链接库链接错误](../../figures/dll_linking_error.png)

解决方法如下所示：

1. 进入“serving-1.15.0/third_party/tf_adapter“目录，执行如下命令。

    在“tf_adapter“文件夹下复制“_tf_adapter.so“文件，并将“_tf_adapter.so“文件名修改为“lib_tf_adapter.so“。

    ```bash
    cp ${TFPLUGIN_INSTALL_PATH}/npu_bridge/_tf_adapter.so .
    mv _tf_adapter.so lib_tf_adapter.so
    ```

2. 执行如下命令清理上次编译的缓存，避免增量编译。

    ```bash
    rm -rvf /opt/tf_serving
    bazel clean
    ```

3. 编译TF Serving。

    在TF Serving安装目录“serving-1.15.0“下执行如下命令，编译TF Serving。

    ```bash
    bazel --output_user_root=/opt/tf_serving build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" tensorflow_serving/model_servers:tensorflow_model_server
    ```

    其中“--output_user_root“参数指定了TF Serving的安装路径。请根据实际进行指定。
