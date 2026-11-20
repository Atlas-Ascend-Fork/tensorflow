# 安装开源框架TensorFlow 2.6.5

## 安装前准备

> [!NOTE]说明
> TensorFlow 2.6.5配套的Python版本是：Python3.7.x（3.7.5~3.7.11）、Python3.8.x、Python3.9.x。

- 对于x86架构，跳过安装前准备。

- 对于aarch64架构。

    由于TensorFlow依赖h5py，而h5py依赖HDF5，需要先编译安装HDF5，否则使用pip安装h5py会报错，以下步骤以root用户操作。

    1. 确保已安装配套版本的Python。
    2. 编译安装HDF5 1.10.5版本。
        1. 访问[下载链接](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz)下载HDF5源码包，并上传到安装环境的任意目录。
        2. 进入源码包所在目录，执行如下命令解压源码包。

            ```bash
            tar -zxvf hdf5-1.10.5.tar.gz
            ```

        3. 进入解压后的文件夹，执行配置、编译和安装命令：

            ```bash
            cd hdf5-1.10.5/
            ./configure --prefix=/usr/local/hdf5
            make -j16 && make install
            ```

        4. 配置环境变量。

            ```bash
            export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
            export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:$LD_LIBRARY_PATH
            ```

    3. 安装h5py。
        1. root用户下执行如下命令安装h5py依赖包。

            ```bash
            pip3 install "Cython<3"
            pip3 install wheel
            ```

        2. root用户下执行如下命令安装h5py。

            ```bash
            pip3 install h5py==3.1.0
            ```

            若在线安装h5py 3.1.0失败，可单击[链接](https://github.com/h5py/h5py/archive/refs/tags/3.1.0.zip)获取源码包，使用源码编译安装：

            ```bash
            unzip h5py-3.1.0.zip
            cd h5py-3.1.0
            python3 setup.py build
            python3 setup.py install
            ```

## 安装TensorFlow

> [!NOTE]说明
> TensorFlow 2.6.5存在漏洞，请参考[相关漏洞及其修复方案](https://nvd.nist.gov/vuln/search/results?isCpeNameSearch=true&query=cpe%3A2.3%3Aa%3Agoogle%3Atensorflow%3A2.6.5%3A*%3A*%3A*%3A*%3A*%3A*%3A*&results_type=overview&form_type=Advanced&startIndex=0)处理。

需要安装TensorFlow才可以进行算子开发以及训练业务开发。

- 对于x86架构：直接从pip源下载即可，系统要求等具体请参考[TensorFlow官网](https://www.tensorflow.org/install/pip?lang=python3)。需要注意TensorFlow官网提供的指导描述有误，从pip源下载CPU版本需要显式指定tensorflow-cpu，如果不指定CPU，默认下载的是GPU版本。安装命令参考如下：

    如下命令如果使用非root用户安装，需要在安装命令后加上**--user**，例如：**pip3 install tensorflow-cpu==2.6.5 --user**

    ```bash
    pip3 install tensorflow-cpu==2.6.5
    ```

- 对于aarch64架构：由于pip源未提供对应的版本，所以需要用户使用官网要求的linux_gcc7.3.0编译器编译TensorFlow 2.6.5，编译步骤参考官网[TensorFlow官网](https://www.tensorflow.org/install/source)。

    在下载完[tensorflow tag v2.6.5](https://github.com/tensorflow/tensorflow/releases/tag/v2.6.5)源码后需要执行如下步骤：

1. 下载“nsync-1.22.0.tar.gz”源码包。
    1. 进入源码目录，打开“tensorflow/workspace2.bzl”文件，找到其中name为nsync的“tf_http_archive”定义。

        ```text
        tf_http_archive(
            name = "nsync",
            sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
            strip_prefix = "nsync-1.22.0",
            system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
            urls = [           "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
                "https://github.com/google/nsync/archive/1.22.0.tar.gz",
            ],
        )
        ```

    2. 从urls中的任一路径下载nsync-1.22.0.tar.gz的源码包，保存到任意路径。

2. 修改“nsync-1.22.0.tar.gz”源码包。
    1. 切换到nsync-1.22.0.tar.gz所在路径，解压缩该源码包。解压缩后存在“nsync-1.22.0”文件夹和“pax_global_header”文件。
    2. 编辑“nsync-1.22.0/platform/c++11/atomic.h”。

        修改NSYNC_CPP_START_后的内容，修改后内容如下。

        ```c
        #include "nsync_cpp.h"
        #include "nsync_atomic.h"

        NSYNC_CPP_START_

        #define ATM_CB_() __sync_synchronize()

        static INLINE int atm_cas_nomb_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
            int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_relaxed, std::memory_order_relaxed));
            ATM_CB_();
            return result;
        }
        static INLINE int atm_cas_acq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
            int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_acquire, std::memory_order_relaxed));
            ATM_CB_();
            return result;
        }
        static INLINE int atm_cas_rel_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
            int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_release, std::memory_order_relaxed));
            ATM_CB_();
            return result;
        }
        static INLINE int atm_cas_relacq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
            int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_acq_rel, std::memory_order_relaxed));
            ATM_CB_();
            return result;
        }
        ```

3. 重新压缩“nsync-1.22.0.tar.gz”源码包。

    将上个步骤中解压出的内容压缩为一个新的“nsync-1.22.0.tar.gz”源码包（例如保存在“/tmp/nsync-1.22.0.tar.gz”）。

4. 重新生成“nsync-1.22.0.tar.gz”源码包的sha256sum校验码。

    执行如下命令后得到sha256sum校验码（一串数字和字母的组合）。

    ```bash
    sha256sum /tmp/nsync-1.22.0.tar.gz
    ```

5. 修改sha256sum校验码和urls。

    进入tensorflow tag源码目录，打开“tensorflow/workspace2.bzl”文件，找到其中name为nsync的“tf_http_archive”定义，其中“sha256=”后面的数字填写步骤4得到的校验码，“urls=”后面的列表第二行，填写存放“nsync-1.22.0.tar.gz”的file://索引。

    ```text
    tf_http_archive(
        name = "nsync",
        sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
        strip_prefix = "nsync-1.22.0",
        system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
            "file:///tmp/nsync-1.22.0.tar.gz",
            "https://github.com/google/nsync/archive/1.22.0.tar.gz",
        ],
    )
    ```

6. 继续参考官方文档（[https://www.tensorflow.org/install/source](https://www.tensorflow.org/install/source)）完成编译。

    > [!NOTE]说明
    > ABI的配置在TensorFlow和FwkPlugin中需要保持一致（即配置为0），否则会导致导入TFA失败。

7. 安装编译好的TensorFlow。

    以上步骤执行完后会打包TensorFlow到指定目录，进入指定目录后执行如下命令安装：

    如下命令如果使用非root用户安装，需要在安装命令后加上**--user**，例如：**pip3 install tensorflow-2.6.5-\*.whl --user**

    ```bash
    pip3 install tensorflow-2.6.5-*.whl
    ```

8. 执行如下命令验证安装效果。

    ```bash
    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```

    如果返回了张量则表示安装成功。安装TensorFlow时会自动重装numpy，导入时如果提示numpy版本不兼容，请参考[安装TensorFlow 2.6.5后，执行import tensorflow时报错](../migration/faq/tensorFlow-2-6-5_import_error.md)重装配套版本的numpy。
