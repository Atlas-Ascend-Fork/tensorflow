# Installing TensorFlow 2.6.5

## Preparing for Installation

> [!NOTE]NOTE
>TensorFlow 2.6.5 matches Python 3.7._x_ \(3.7.5–3.7.11\), Python 3.8._x_, and Python 3.9._x_.

- For the x86 architecture, skip the installation preparations.

- For the AArch64 architecture:

    TensorFlow depends on h5py, and h5py depends on HDF5. Therefore, you need to compile and install HDF5 first. Otherwise, an error is reported when you use pip to install h5py. Perform the following operations as the  **root**  user.

    1. Ensure that Python of a matching version has been installed.
    2. Compile and install HDF5 1.10.5.
        1. [Download](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz)  the HDF5 source package and upload it to any directory in the installation environment.
        2. Go to the directory where the source package is stored, and run the following command to decompress the source package:

            ```bash
            tar -zxvf hdf5-1.10.5.tar.gz
            ```

        3. Go to the decompressed folder and run the following configuration, build, and installation commands:

            ```bash
            cd hdf5-1.10.5/
            ./configure --prefix=/usr/local/hdf5
            make -j16 && make install
            ```

        4. Set the environment variable:

            ```bash
            export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
            export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:$LD_LIBRARY_PATH
            ```

    3. Install h5py.
        1. Run the following command as the  **root**  user to install the h5py dependency package:

            ```bash
            pip3 install "Cython<3"
            pip3 install wheel
            ```

        2. Run the following command as the  **root**  user to install h5py:

            ```bash
            pip3 install h5py==3.1.0
            ```

            If h5py 3.1.0 fails to be installed online, click  [here](https://github.com/h5py/h5py/archive/refs/tags/3.1.0.zip)  to obtain the source package, and use the source code to compile and install h5py 3.1.0:

            ```bash
            unzip h5py-3.1.0.zip
            cd h5py-3.1.0
            python3 setup.py build
            python3 setup.py install
            ```

## Installing TensorFlow

> [!NOTE]NOTE
>TensorFlow 2.6.5 has vulnerabilities. For details about how to handle the vulnerabilities, see  [Vulnerabilities and Fixing Solutions](https://nvd.nist.gov/vuln/search/results?isCpeNameSearch=true&query=cpe%3A2.3%3Aa%3Agoogle%3Atensorflow%3A2.6.5%3A*%3A*%3A*%3A*%3A*%3A*%3A*&results_type=overview&form_type=Advanced&startIndex=0).

TensorFlow must be installed to develop operators and training services.

- For the x86 architecture, download the software package from the pip source. For details about system requirements, see the  [TensorFlow official website](https://www.tensorflow.org/install/pip?lang=python3&hl=en). Note that the instructions provided by the TensorFlow website are incorrect. To download the CPU version from the pip source, you need to explicitly specify  **tensorflow-cpu**. Otherwise, the GPU version is downloaded by default. The installation command is as follows.

    If you run the following command as a non-root user, add  **--user**  to the end of the installation command, for example,  **pip3 install tensorflow-cpu==2.6.5 --user**.

    ```bash
    pip3 install tensorflow-cpu==2.6.5
    ```

- For the AArch64 architecture, the pip source does not provide the corresponding version. Therefore, you need to use  **linux_gcc7.3.0**  to compile TensorFlow 2.6.5. For details about the compilation procedure, see the  [TensorFlow official website](https://www.tensorflow.org/install/source).

    After downloading the  [tensorflow tag v2.6.5](https://github.com/tensorflow/tensorflow/releases/tag/v2.6.5)  source code, perform the following steps:

1. Download the  **nsync-1.22.0.tar.gz**  source package.
    1. Go to the source code directory, open the  **tensorflow/workspace2.bzl**  file, and find the  **tf_http_archive**  definition whose  **name**  is  **nsync**.

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

    2. Download the  **nsync-1.22.0.tar.gz**  source package from any path in  **urls**  and save it to any path.

2. Modify the  **nsync-1.22.0.tar.gz**  source package.
    1. Go to the directory where  **nsync-1.22.0.tar.gz**  is stored and decompress the source package. Find the decompressed  **nsync-1.22.0**  folder and the  **pax_global_header**  file.
    2. Edit the  **nsync-1.22.0/platform/c++11/atomic.h**  file.

        Append the following information in bold to  **NSYNC_CPP_START_**.

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

3. Compress the  **nsync-1.22.0.tar.gz**  source package.

    Compress the content obtained in the previous step into a new source package  **nsync-1.22.0.tar.gz**  and save it to a directory, for example,  **/tmp/nsync-1.22.0.tar.gz**.

4. Generate a  **sha256sum**  checksum for the  **nsync-1.22.0.tar.gz**  source package.

    The obtained  **sha256sum**  checksum is a combination of digits and letters.

    ```bash
    sha256sum /tmp/nsync-1.22.0.tar.gz
    ```

5. Change the  **sha256sum**  checksum and  **urls**.

    Go to the  **tensorflow tag**  source code directory, open the  **tensorflow/workspace2.bzl**  file, find the definition of  **tf_http_archive**  whose  **name**  is  **nsync**, enter the verification code obtained in  step4 after  **sha256=**, enter the first line of the list after  **urls=**, and enter the  **file://**  index for storing the  **nsync-1.22.0.tar.gz**  file.

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

6. Complete the compilation by referring to the official document \([https://www.tensorflow.org/install/source](https://www.tensorflow.org/install/source)\).

    > [!NOTE]NOTE
    >The ABI configuration in TensorFlow and FwkPlugin must be the same \(that is, set to  **0**\). Otherwise, the TFA fails to be imported.

7. Install the compiled TensorFlow.

    After the preceding steps are complete, TensorFlow is packaged to the specified directory. Go to the specified directory and run the following installation command.

    If you run the following command as a non-root user, add  **--user**  to the end of the installation command, for example,  **pip3 install tensorflow-2.6.5-\*.whl --user**.

    ```bash
    pip3 install tensorflow-2.6.5-*.whl
    ```

8. Run the following command to verify the installation:

    ```bash
    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```

    If a tensor is returned, the installation is successful. NumPy is automatically reinstalled when you install TensorFlow. If a message is displayed indicating that the NumPy version is incompatible, reinstall NumPy of the matching version manually by referring to  [An Error Is Reported When import tensorflow Is Executed After TensorFlow 2.6.5 Is Installed](../migration/faq/tensorFlow-2-6-5_import_error.md).
