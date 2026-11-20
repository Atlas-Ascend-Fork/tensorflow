# FAQs

## What Should I Do If the swig Dependency Package Cannot Be Installed?

### Symptom

If the swig dependency package fails to be installed during TF Serving build, the error message shown below is displayed:

```text
E: The package ascend-cann-toolkit needs to be reinstalled, but I can't  find an archive for it.
```

### Solution

Perform the following steps to reinstall the swig dependency package:

1. Back up  **/var/lib/dpkg/status**.

    ```bash
    cp /var/lib/dpkg/status /{newfilepath}/status
    ```

    In the command,  _\{newfilepath\}_  indicates the path to be backed up. Replace it with the actual path.

2. Open the  **/var/lib/dpkg/status**  file, locate the record of the software package that fails to install, and remove the record, as shown in the marked position in  the following figure.

    ```bash
    vim /var/lib/dpkg/status
    ```

    ![](../../figures/swing_error.png "software-package-record")

3. Reinstall swig.

    ```bash
    apt install swig
    ```

## What Should I Do If an Error About builtins Missing Is Displayed During TF Serving Build?

### Symptom

During TF Serving compilation, the  **builtins**  dependency module may fail to be queried, as shown in  the following figure.

![](../../figures/builtins_loss_error.png "error-about-builtins-missing")

### Solution

The reason is that the  **future**  dependency package is missing. The solution is to install the  **future**  dependency package and check the Python direction.

1. Install the  **future**  dependency package.

    ```bash
    pip3.7 install future
    ```

2. Check whether the  **Python**  soft link is pointing to  **Python 3.7.5**.

    Python 3.7.5 is required for TF Serving compilation, and TF Serving-related scripts use the Python interpreter keyword  **Python**  by default. However, this keyword is different from that used by Python 2.7, to which the  **Python**  soft link is pointing by default. Therefore, you need to redirect the  **Python**  soft link to point to  **Python 3.7.5**.

    1. Check the Python version to which the  **Python**  soft link is pointing.

        ```bash
        python --version
        ```

        If the soft link is pointing to  **Python 3.7.5**, you can directly recompile TF Serving. Otherwise, go to the next step.

    2. Create a  **Python**  soft link pointing to  **Python 3.7.5**.

        ```bash
        ln -sf /usr/local/python3.7.5/bin/python3.7  /usr/bin/python
        ```

3. Perform TF Serving build again.
4. (Optional) Restore the  **Python**  soft link.

    ```bash
    ln -sf /usr/bin/python2 /usr/bin/python
    ```

## What Should I Do If the tfadapter.tar File Is Not Generated After TF Adapter Build Is Complete?

### Symptom

The system displays a message indicating that the compilation is successful, but the  **tfadapter.tar**  package is not generated in the  **output**  directory.

### Solution

The solution is to modify the  **tf_adapter_2.x/CI_Build**  file and perform recompilation.

1. In the directory where the said file resides, run the following command to open the file:

    ```bash
    vim tf_adapter_2.x/CI_Build
    ```

    The modifications are as follows:

    ```text
    CONFIGURE_DIR=$(dirname "$0")
    cd "${CONFIGURE_DIR}"

    ############## NPU modify begin #############
    if [ "$(arch)" != "xxx" ];then
      mkdir -p build/dist/python/dist/
      touch build/dist/python/dist/npu_device-0.1-py3-none-any.whl
      exit 0
    fi
    ############## NPU modify end ###############

    # The source code is as follows:
    if [ "$(arch)" != "x86_64" ];then
      mkdir -p build/dist/python/dist/
      touch build/dist/python/dist/npu_device-0.1-py3-none-any.whl
      exit 0
    fi
    ```

2. Perform TF Adapter build again.
