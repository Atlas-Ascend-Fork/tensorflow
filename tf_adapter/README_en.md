# Huawei Ascend Adapter for TF1.X

## Introduction

Ascend Adapter for TF1.X provides NPU computing capabilities to developers who use the TensorFlow framework.

Install the TF Adapter plugin and add a few configuration options to your existing TensorFlow scripts to accelerate your training tasks on NPUs.

![tfadapter1](../docs/en/figures/tfadapter1.png)

The left side of the preceding figure shows the TensorFlow 1.15 framework architecture, and the right side shows the TF Adapter architecture. Each layer of the TensorFlow framework has a corresponding implementation in TF Adapter.

## Compilation and Installation

You can build the TF Adapter software package from the source code in this repository and deploy it in an NPU environment.

### Prerequisites

The Ascend Adapter software package must be compiled in a Linux OS environment. Install the following software dependencies:

- **Python 3.7**

  Ascend Adapter requires Python 3.7 for compilation.

- **TensorFlow 1.15.0**

  Ascend Adapter has strict version compatibility with TensorFlow. Before building the TF Adapter software package from source, ensure that you have correctly installed [TensorFlow v1.15.0](https://www.tensorflow.org/install/pip). For installation instructions, refer to the "TensorFlow 1.15 Model Migration > Prerequisites > Install open-source framework TensorFlow 1.15" section in the [Ascend Community Documentation - TensorFlow 1.15 Model Migration](https://hiascend.com/document/redirect/canntfmigr).

- **GCC >= 7.3.0**

  Ascend Adapter requires GCC 7.3.0 or later for compilation.

- **CMake >= 3.14.0**

  Ascend Adapter requires CMake 3.14.0 or later for compilation.

- **SWIG**

  Ascend Adapter source code compilation depends on SWIG. Install SWIG using the following command:

  ```shell
  # Install on Ubuntu/Debian. Install on other operating systems as needed.
  apt-get install swig
  ```

- **CANN Development Kit (cann-toolkit)**

  Obtain the corresponding CANN version and download and install `Ascend-cann-toolkit_<cann_version>_linux-<arch>.run` from the [CANN Download Page](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/).

  Install the CANN Development Kit using the following command:

  ```bash
  # Install command (--install-path is optional)
  bash Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install --quiet --install-path=${install_path}
  ```

  - `<cann_version>`: CANN package version number.
  - `<arch>`: Operating system architecture, such as `x86_64` or `aarch64`.
  - `${install_path}`: Installation path. The default installation path is `/usr/local/Ascend`.

  For detailed CANN installation instructions, refer to the [CANN Software Installation Guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware).

### Download the Source Code

```bash
git clone https://gitcode.com/cann/tensorflow.git
cd tensorflow
```

### Build the TF Adapter Source Code

Run the following command to compile the TF Adapter source code:

```bash
bash tf_adapter/build.sh -c
```

> **Precautions:** Before running the build command, ensure that the following environment variables are configured:

1. Configure the CANN Development Kit environment variables:

   ```bash
   # Default installation path. The following uses root as an example, For non-root users, replace /usr/local with ${HOME}.
   source /usr/local/Ascend/cann/set_env.sh
   # Custom installation path
   source ${install_path}/cann/set_env.sh
   ```

After compilation, the TF Adapter installation package is generated at the following path:

```bash
./build/tfadapter/dist/python/dist/npu_bridge-1.15.0-py3-none-manylinux2014_<arch>.whl
```

`<arch>` indicates the operating system architecture. The value can be `x86_64` or `aarch64`.

### Run UT/ST

**Prerequisites:**

- Ensure that the `lcov` tool is correctly installed.
- The `gcc` and `gcov` on the compilation environment must be of matching versions.

Run the following command to execute UT:

```bash
bash tf_adapter/build.sh -u
```

Run the following command to execute ST:

```bash
bash tf_adapter/build.sh -s
```

After UT/ST execution is complete, check the test results in the output logs. A successful test case prints `passed` with no `failed` output, confirming that all test cases have passed.

### Install TF Adapter

Run the following command to install TF Adapter. Replace the package name with the actual one.

```bash
pip3 install ./build/tfadapter/dist/python/dist/npu_bridge-1.15.0-py3-none-manylinux2014_<arch>.whl --upgrade
```

After installation, the TF Adapter files are installed to the Python interpreter search path, such as `/usr/local/python3.7.5/lib/python3.7/site-packages`. The installed folders are `npu_bridge` and `npu_bridge-1.15.0.dist-info`.

> [!NOTE]
> To uninstall the TF Adapter software package, run the following command:
>
> `pip3 uninstall -y npu_bridge`

## FAQ

### 1. Running ./build.sh prompts to configure the SWIG path

Run the following command to install SWIG:

```bash
pip3 install swig
```

### 2. Running ./build.sh on Ubuntu displays "Could not import the lzma module"

Run the following command to install lzma:

`apt-get install liblzma-dev`

**Precautions:** This dependency must be installed before Python installation. If your operating system already has a compatible Python environment installed, you must recompile the Python environment after installing liblzma-dev.

### 3. TensorFlow Source Code Customization (Optional)

In some scenarios, you may want to use a customized or modified TensorFlow with the TF Adapter software package. Because TF Adapter links to the source code from the official TensorFlow website by default, symbol mismatches may cause core dumps when using the TF Adapter software package. To enable TF Adapter to work with your customized TensorFlow source code, modify the `tensorflow/cmake/tensorflow.cmake` file in the TF Adapter source code as follows:

![Before modification: TF Adapter links to the official TensorFlow source code](../docs/en/figures/tensorflow_cmake.png)

Modify the URL and URL_HASH MD5 under FetchContent_Declare in the figure to the address and MD5 value of the TensorFlow package in your environment.

For example, if your TensorFlow package is placed in the `/opt/hw` path, modify the tensorflow.cmake source code as follows:

![After modification: TF Adapter links to your customized TensorFlow source code](../docs/en/figures/revise_tensorflow.png)

### 4. TF Adapter Source Code Customization (Optional)

If you want to modify the TF Adapter source code, such as adding link paths or linking other shared libraries, modify the `tensorflow/CMakeLists.txt` file in the TF Adapter source code. Modify the compilation configuration under the `ENABLE_OPEN_SRC` branch to apply the changes.

![CMakeLists.txt file](../docs/en/figures/cmake.png)
