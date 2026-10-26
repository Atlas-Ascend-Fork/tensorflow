# Huawei Ascend Adapter for TF2.X

## Introduction

Ascend Adapter for TF2.X provides NPU computing capabilities to developers who use the TensorFlow 2.x framework. Install the Ascend Adapter for TF2.X plugin and add a few configuration options to your existing TensorFlow 2.x scripts to accelerate your training tasks on NPUs.

![tfadapter2](../docs/en/figures/tfadapter2.png)

## Compilation and Installation

You can build the Ascend Adapter software package from the source code and install it on an Ascend AI processor environment.

### Prerequisites

The Ascend Adapter software package must be compiled in a Linux OS environment. Install the following software dependencies:

- **Python 3.7 to Python 3.9**

  Ascend Adapter can be compiled using Python 3.7, Python 3.8, or Python 3.9.

- **TensorFlow 2.6.5**

  Ascend Adapter has strict version compatibility with TensorFlow. Before building from source, ensure that you have correctly installed [TensorFlow v2.6.5](https://www.tensorflow.org/install). For installation instructions, refer to the "TensorFlow 2.6.5 Model Migration > Prerequisites > Install open-source framework TensorFlow 2.6.5" section in the [Ascend Community Documentation - TensorFlow 2.6.5 Model Migration](https://hiascend.com/document/redirect/canntfmigr).

- **GCC >= 7.3.0**

  Ascend Adapter requires GCC 7.3.0 or later for compilation.

- **CMake >= 3.14.0**

  Ascend Adapter requires CMake 3.14.0 or later for compilation.

- **SWIG >= 4.1.0**

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
cd tensorflow/tf_adapter_2.x
```

### Build the Source Code

```bash
bash build.sh -c
```

> **Precautions:** Before running the build command, ensure that the following environment variables are configured:

1. Configure the CANN Development Kit environment variables:

   ```bash
   # Default installation path. The following uses root as an example.
   # For non-root users, replace /usr/local with ${HOME}.
   source /usr/local/Ascend/cann/set_env.sh
   # Custom installation path
   source ${install_path}/cann/set_env.sh
   ```

After compilation, the installation package is generated at:

```bash
./build/dist/python/dist/npu_device-2.6.5-py3-none-manylinux2014_<arch>.whl
```

`<arch>` indicates the operating system architecture. The value can be `x86_64` or `aarch64`.

### Run UT/ST

**Prerequisites:**

- Ensure that the `lcov` tool is correctly installed.
- The `gcc` and `gcov` on the compilation environment must be of matching versions.

Run the following command to execute UT:

```bash
bash build.sh -u
```

Run the following command to execute ST:

```bash
bash build.sh -s
```

After UT/ST execution is complete, check the test results in the output logs. A successful test case prints `passed` with no `failed` output, confirming that all test cases have passed.

### Install TF Adapter

Run the following command to install TF Adapter. Replace the package name with the actual one.

```bash
pip3 install ./build/dist/python/dist/npu_device-2.6.5-py3-none-manylinux2014_<arch>.whl --upgrade
```

> [!NOTE]
> To uninstall the TF Adapter software package, run the following command:
>
> `pip3 uninstall -y npu_device`
