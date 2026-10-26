# Huawei Ascend Adapter for TF2.X

## 简介

Ascend Adapter for TF2.X 致力于将NPU运算能力便捷地提供给使用Tensorflow 2.x框架的开发者。开发者只需安装Ascend Adapter for
TF2.X插件，并在现有Tensorflow 2.x脚本中添加少量配置，即可实现在NPU上加速自己的训练任务。

![tfadapter2](../docs/zh/figures/tfadapter2.png)

## 编译与安装

您可以从源代码构建 Ascend Adapter 软件包并将其安装在昇腾AI处理器环境上。

### 环境准备

Ascend Adapter 软件包需要在Linux OS环境上进行编译，同时环境上需要安装一下软件依赖:

- **Python3.7~Python3.9**

  Ascend Adapter可以使用python3.7、Python3.8、Python3.9版本进行编译。

- **TensorFlow 2.6.5**

  Ascend Adapter 与 Tensorflow 有严格的匹配关系，从源码构建前，您需要确保已经正确安装了[Tensorflow v2.6.5 版本](https://www.tensorflow.org/install) ，安装方式可参见[昇腾社区文档中心-TensorFlow 2.6.5模型迁移](https://hiascend.com/document/redirect/canntfmigr)中的“TensorFlow 2.6.5模型迁移 > 环境准备 > 安装开源框架TensorFlow 2.6.5”章节。

- **GCC >= 7.3.0**

  Ascend Adapter 需要使用7.3.0及更高版本的gcc编译。

- **CMake >= 3.14.0**

  Ascend Adapter 需要使用3.14.0及更高版本的cmake编译。

- **SWIG >= 4.1.0**

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
cd tensorflow/tf_adapter_2.x
```

### 执行编译

```bash
bash build.sh -c
```

> 请注意：执行编译命令前，请确保环境中已配置了以下环境变量：

1. 配置CANN开发套件包的环境变量：

   ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/cann/set_env.sh
   # 指定路径安装
   source ${install_path}/cann/set_env.sh
   ```

编译结束后，安装包会生成在

```bash
./build/dist/python/dist/npu_device-2.6.5-py3-none-manylinux2014_<arch>.whl
```

\<arch>表示操作系统架构，取值为x86_64与aarch64。

### 执行UT/ST

执行如下命令运行UT：

**前置条件**：

- 确保 `lcov` 工具已正确安装。
- 编译运行环境上的 `gcc` 和 `gcov` 必须是配套版本。

```bash
bash build.sh -u
```

执行如下命令运行ST：

```bash
bash build.sh -s
```

UT/ST执行完成后，可根据输出日志查看测试执行情况。用例执行成功会打印`passed`，且无`failed`打印，确认所有测试用例通过。

### 安装TF Adapter

执行如下命令安装TF Adapter，请注意替换为实际的包名。

```bash
pip3 install ./build/dist/python/dist/npu_device-2.6.5-py3-none-manylinux2014_<arch>.whl --upgrade
```

> [!NOTE]说明
> 若您需要卸载TF Adapter软件包，可以执行如下命令：
>
> `pip3 uninstall -y npu_device`
