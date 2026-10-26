# Huawei Ascend Adapter for TF1.X

## 简介

Ascend Adapter for TF1.X 致力于将NPU的运算能力便捷地提供给使用Tensorflow框架的开发者。

开发者只需安装TF Adapter插件，并在现有TensorFlow脚本中添加少量配置，即可实现在NPU上加速自己的训练任务。

![tfadapter1](../docs/zh/figures/tfadapter1.png)

上图左侧是TensorFlow 1.15框架架构示例，右侧为TF Adapter架构示例，可以看出TensorFlow框架的每一层在TF Adapter中都有对应的实现。

## 编译与安装

您可以通过此仓中的源代码构建TF Adapter软件包并将其部署在NPU环境上。

### 环境准备

Ascend Adapter 软件包需要在Linux OS环境上进行编译，同时环境上需要安装一下软件依赖:

- **Python3.7**

  Ascend Adapter 需要使用python3.7版本进行编译。

- **TensorFlow 1.15.0**

  Ascend Adapter 与 Tensorflow 有严格的匹配关系，通过源码构建TF Adapter软件包前，您需要确保已经正确安装了 [Tensorflow v1.15.0版本](https://www.tensorflow.org/install/pip) ，安装方式可参见[昇腾社区文档中心-TensorFlow 1.15模型迁移](https://hiascend.com/document/redirect/canntfmigr)中的“TensorFlow 1.15模型迁移 > 环境准备 > 安装开源框架TensorFlow 1.15”章节。

- **GCC >= 7.3.0**

  Ascend Adapter 需要使用7.3.0及更高版本的gcc编译

- **CMake >= 3.14.0**

  Ascend Adapter 需要使用3.14.0及更高版本的cmake编译

- **SWIG**

  Ascend Adapter 源码编译依赖SWIG， SWIG安装命令示例如下：

  ```shell
  # Ubuntu/Debian操作系统安装命令示例如下，其他操作系统请自行安装
  apt-get install swig
  ```

- **CANN开发套件包（cann-toolkit）**

  请获取对应的CANN软件版本号，并在“[CANN下载页面](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)”下载并安装对应版本的`Ascend-cann-toolkit_<cann_version>_linux-<arch>.run`。

  CANN开发套件包（cann-toolkit）安装命令示例如下：

  ```bash
  # 安装命令（其中--install-path为可选）
  bash Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install --quiet --install-path=${install_path}
  ```

  - `<cann_version>`：表示CANN包版本号。
  - `<arch>`：表示操作系统架构，如`x86_64`、`aarch64`。
  - `${install_path}`：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

  CANN软件更详细的安装方法可参见[CANN软件安装指南](https://hiascend.com/document/redirect/CannCommunityInstSoftware)。

### 源码下载

```bash
git clone https://gitcode.com/cann/tensorflow.git
cd tensorflow
```

### 编译TF Adapter源码生成安装包

执行如下命令，对TF Adapter源码进行编译：

```bash
bash tf_adapter/build.sh -c
```

> 请注意：执行编译命令前，请确保环境中已配置了以下环境变量：

1. 配置CANN开发套件包的环境变量：

   ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/cann/set_env.sh
   # 指定路径安装
   source ${install_path}/cann/set_env.sh
   ```

编译结束后，TF Adapter安装包生成在如下路径：

```bash
./build/tfadapter/dist/python/dist/npu_bridge-1.15.0-py3-none-manylinux2014_<arch>.whl
```

\<arch>表示操作系统架构，取值为x86_64与aarch64。

### 执行UT/ST

**前置条件**：

- 确保 `lcov` 工具已正确安装。
- 编译运行环境上的 `gcc` 和 `gcov` 必须是配套版本。

执行如下命令运行UT：

```bash
bash tf_adapter/build.sh -u
```

执行如下命令运行ST：

```bash
bash tf_adapter/build.sh -s
```

UT/ST执行完成后，可根据输出日志查看测试执行情况。用例执行成功会打印`passed`，且无`failed`打印，确认所有测试用例通过。

### 安装TF Adapter

执行如下命令安装TF Adapter，请注意替换为实际的包名。

```bash
pip3 install ./build/tfadapter/dist/python/dist/npu_bridge-1.15.0-py3-none-manylinux2014_<arch>.whl --upgrade
```

执行完成后，TF Adapter相关文件安装到python解释器搜索路径下，例如“/usr/local/python3.7.5/lib/python3.7/site-packages”路径，安装后文件夹为“npu_bridge”与“npu_bridge-1.15.0.dist-info”。

> [!NOTE]说明
> 若您需要卸载TF Adapter软件包，可以执行如下命令：
>
> `pip3 uninstall -y npu_bridge`

## FAQ

### 1. 执行./build.sh时提示配置swig的路径

需要执行以下命令安装swig

```bash
pip3 install swig
```

### 2. Ubuntu系统中执行./build.sh时提示“Could not import the lzma module”

​     执行如下命令进行lzma的安装：

​     `apt-get install liblzma-dev`

​      需要注意，此依赖需要在Python安装之前安装，如果用户操作系统中已经安装满足要求的Python环境，在此之后再安装liblzma-dev，则需要重新编译Python环境。

### 3. TensorFlow源码定制（可选）

在部分场景下，您可能会把自己定制或者修改过的TensorFlow与TF Adapter软件包配合使用，由于TF Adapter默认链接的是TensorFlow官方网站的源码，因此您在使用TF Adapter软件包的时候，可能会因为符号不匹配而出现coredump问题。为了使TF Adapter能适配您的TensorFlow源码，您需要将TF Adapter源码下的tensorflow/cmake/tensorflow.cmake文件稍作修改，详细修改点如下：

![修改前TF_Adapter链接的是tensorflow官网源码](../docs/zh/figures/tensorflow_cmake.png)

修改图中FetchContent_Declare下的URL和URL_HASH MD5，将其替换成您自己环境上的tensorflow软件包的地址和MD5值。

例如，您的tensorflow软件包如果放在/opt/hw路径下，则您此处tensorflow.cmake的源码可以修改为

![修改后TF_Adapter链接您环境上的tensorflow定制源码](../docs/zh/figures/revise_tensorflow.png)

### 4. TF Adapter源码定制（可选）

如果您想对TF Adapter的源码进行修改，比如添加链接路径，或链接其他so等操作，您可以修改TF Adapter源码下的tensorflow/CMakeLists.txt文件，只需要将ENABLE_OPEN_SRC分支下的编译配置做修改，便可以生效

![CMakeList.txt文件](../docs/zh/figures/cmake.png)
