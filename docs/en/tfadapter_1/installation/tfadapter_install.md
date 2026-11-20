# Installing the Framework Plugin Package

Ascend provides the adaptation plugin TF Adapter for interconnecting with the deep-learning framework TensorFlow. If you need to perform training or online inference of the TensorFlow network on NPUs, you need to install the TF Adapter package.

- If you install the plugin package for the first time, see [Installing the Plugin Package](#installing-the-plugin-package).
- If you have installed the package, upgrade it to the latest version. For details, see [Upgrading the Plugin Package](#upgrading-the-plugin-package).

## Installing the Plugin Package

1. Download the TF Adapter installation package.
    1. Access the [TF Adapter GitCode](https://gitcode.com/cann/tensorflow/tags)  repository to go to the TF Adapter release tag page.

    2. Select a matched CANN version and click  **Release**  to go to the  **Release**  page.

        TF Adapter repository tags follow the naming convention:  **tfa_$\{_Tag version_\}_$\{_TF Adapter version_\}**.  _**$\{TF Adapter version\}**_  is the same as the matched CANN version.

    3. Obtain the .whl installation package of TF Adapter on the  **Release**  page.

        For TensorFlow 1.15, download  **npu_bridge-1.15.0-py3-none-manylinux2014__<arch\>_.whl**.

        _**<arch\>**_  indicates the OS architecture, which can be  **x86_64**  or  **aarch64**.

2. Install the TF Adapter package.
    1. Log in to the installation environment as the installation user and upload the obtained package to any path \(for example,  **/home/package**\) in the installation environment.
    2. Run the following command to install TF Adapter:

        ```bash
        pip3 install "npu_bridge-1.15.0-py3-none-manylinux2014_<arch>.whl" --force-reinstall -t "${TFPLUGIN_INSTALL_PATH}"
        ```

        - Replace  **"npu_bridge-1.15.0-py3-none-manylinux2014_<arch\>.whl"**  with the actual package name.
        - **--force-reinstall**: forcibly reinstalls the specified software package.
        - $\{TFPLUGIN_INSTALL_PATH\}  specified by  **-t**  is the installation path of the TF Adapter package, for example,  **"$HOME/Ascend/tfplugin"**.

        The following lists some common pip command options. For more options, run the  **pip3 --help**  command.

        - **--disable-pip-version-check**: shields the pip version check to prevent pip upgrade alarms.
        - **--upgrade**: upgrades the specified software package.
        - **--no-deps**: ignores the dependencies of the installed software package.

3. Set the TF Adapter environment variables so that you can directly use the Python library of TF Adapter.

    ```bash
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
    ```

    $\{TFPLUGIN_INSTALL_PATH\}  is the installation path of the TF Adapter package.

## Upgrading the Plugin Package

To upgrade the TF Adapter package, you need to uninstall the old package and then install the new package.

1. Uninstall the TF Adapter package.
    - **Scenario 1**: TF Adapter 8.0.0 is packaged in .whl format for the first time. If a version earlier than TF Adapter 8.0.0 is installed in your environment, perform the following steps to uninstall it.
        1. Access the directory where the uninstallation script is stored.

            ```bash
            cd /usr/local/Ascend/tfplugin/latest/script
            ```

            **/usr/local/Ascend**  indicates the installation path of the TF Adapter package of a version earlier than 8.0.0. Replace it with the actual path.

        2. Run the  **uninstall.sh**  script to uninstall the TF Adapter package of an earlier version.

            ```bash
            ./uninstall.sh
            ```

    - **Scenario 2**: If a .whl package of TF Adapter 8.0.0 or later is installed, uninstall it by referring to  [Uninstalling the Plugin Package](#uninstalling-the-plugin-package).

2. After the uninstallation is successful, install the TF Adapter .whl package of the new version by referring to  [Installing the Plugin Package](#installing-the-plugin-package).

## Uninstalling the Plugin Package

To uninstall the TF Adapter package, run the following commands:

```bash
## Set the environment variables to specify the path of the Python library of TF Adapter.
export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

# Uninstall the adaptation plugin package of TensorFlow 1.15.
pip3 uninstall -y npu_bridge
```
