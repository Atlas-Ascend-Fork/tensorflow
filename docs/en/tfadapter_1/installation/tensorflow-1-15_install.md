# Installing TensorFlow 1.15

## Installation Preparations

> [!NOTE]NOTE
> TensorFlow 1.15 matches Python 3.7._x_  \(3.7.5 to 3.7.11\).

- For the x86 architecture, skip the installation preparations.
- For the AArch64 architecture,  TensorFlow  depends on h5py, which in turn depends on HDF5. Therefore, you must compile and install HDF5 first. Otherwise, using pip to install h5py will result in an error. Perform the following steps as the  **root**  user.
    1. Ensure that Python of a matching version has been installed.
    2. Compile and install HDF5 1.10.5.
        1. [Download](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz)  the HDF5 source package and upload it to any directory in the installation environment.
        2. Go to the directory where the source package is stored, and run the following command to decompress the source package:

            ```bash
            tar -zxvf hdf5-1.10.5.tar.gz
            ```

        3. Go to the decompressed folder and run the following configuration, compilation, and installation commands:

            ```bash
            cd hdf5-1.10.5/
            ./configure --prefix=/usr/local/hdf5
            make -j16 && make install
            ```

        4. Set environment variables.

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
            pip3 install numpy    # If NumPy is not installed, online installation of h5py 2.8.0 will fail.
            pip3 install h5py==2.8.0
            ```

## Installing TensorFlow

TensorFlow is necessary to develop and verify operators and develop training services.

- For the x86 architecture, download the software package from the pip source. For details about system requirements, see the  [TensorFlow official website](https://www.tensorflow.org/install/pip?lang=python3&hl=en). Note that the instructions provided by the TensorFlow website are incorrect. To download the CPU version from the pip source, you need to explicitly specify  **tensorflow-cpu**. Otherwise, the GPU version is downloaded by default. The installation command is as follows:

  If you run the following command as a non-root user, add  **--user**  to the end of the installation command, for example,  **pip3 install tensorflow-cpu==1.15 --user**.

  Install TensorFlow 1.15.

  ```bash
  pip3 install tensorflow-cpu==1.15
  ```

- For the AArch64 architecture, the pip source does not provide the corresponding version. Therefore, you need to use  **linux_gcc7.3.0**  to compile TensorFlow 1.15. For details about the compilation procedure, see the  [TensorFlow official website](https://www.tensorflow.org/install/source). Pay attention to the following:

After downloading  [tensorflow tag v1.15.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0), perform the following steps:

1. Download the  **nsync-1.22.0.tar.gz**  source package.
    1. Go to the source code directory, open the  **tensorflow/workspace.bzl**  file, and find the  **tf_http_archive**  definition whose  **name**  is  **nsync**.

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

        Before modification:

        ```cpp
        #include "nsync_cpp.h"
        #include "nsync_atomic.h"

        NSYNC_CPP_START_

        static INLINE int atm_cas_nomb_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
                return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                                                     std::memory_order_relaxed, std::memory_order_relaxed));
        }
        static INLINE int atm_cas_acq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
                return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                                                     std::memory_order_acquire, std::memory_order_relaxed));
        }
        static INLINE int atm_cas_rel_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
                return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                                                     std::memory_order_release, std::memory_order_relaxed));
        }
        static INLINE int atm_cas_relacq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
                return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                                                     std::memory_order_acq_rel, std::memory_order_relaxed));
        }
        ```

        After modification:

        ```cpp
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

    Compress the content extracted in the previous step into a new  **nsync-1.22.0.tar.gz**  source package and save it to a path such as  **/tmp/nsync-1.22.0.tar.gz**.

4. <a id="step4"></a>Generate a  **sha256sum**  checksum for the  **nsync-1.22.0.tar.gz**  source package.

    The obtained  **sha256sum**  checksum is a combination of digits and letters.

    ```bash
    sha256sum /tmp/nsync-1.22.0.tar.gz
    ```

5. Change the  **sha256sum**  checksum and  **urls**.

    Go to the TensorFlow source code directory of the corresponding tag, open the  **tensorflow/workspace.bzl**  file, and locate the definition of  **tf_http_archive**  whose  **name**  is  **nsync**. Enter the checksum obtained in  [4](#step4)  after  **sha256=**  and replace the second entry in the list after  **urls=**  with the  **file://**  URI pointing to the location of  **nsync-1.22.0.tar.gz**.

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

    After the preceding steps are complete, TensorFlow is packaged to the specified directory. Go to the specified directory and run the following installation command:

    If you run the following command as a non-root user, add  **--user**  to the end of the installation command, for example,  **pip3 install tensorflow-1.15.0-\*.whl --user**.

    ```bash
    pip3 install tensorflow-1.15.0-*.whl
    ```

8. Run the following command to verify the installation:

    ```bash
    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```

    If a tensor is returned, the installation is successful.
