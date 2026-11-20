# 安装框架插件包TF Adapter

昇腾提供了对接深度学习框架TensorFlow的适配插件TF Adapter，若开发者需要在NPU上执行TensorFlow网络的训练或在线推理，需要安装TensorFlow框架适配插件包TF Adapter。

- 若您是首次安装框架插件包，请参见[安装插件包](#安装插件包)。
- 若您已经安装了框架插件包，需要升级到新版本框架插件包，请参见[升级插件包](#升级插件包)。

## 安装插件包

1. 获取TF Adapter安装包。
    1. 单击[TF Adapter Gitcode仓](https://gitcode.com/cann/tensorflow/tags)，进入TF Adapter发布标签页面。
    2. 选择配套CANN版本的标签，然后单击“发行版”，进入“发行版”页面。

        TF Adapter源码仓标签命名规则为：tfa_\$\{tag版本\}_\$\{TF Adapter软件版本\}，其中$\{TF Adapter软件版本\}与配套的CANN软件版本号一致。

    3. 在“发行版”页面中获取TF Adapter的whl安装包。

        针对TensorFlow 2.6.5框架，您需要获取**“npu_device-2.6.5-py3-none-manylinux2014_<arch\>.whl”**软件包。

        <arch\>表示操作系统架构，取值为x86_64与aarch64。

2. 安装TF Adapter软件包。
    1. 以软件包的安装用户登录安装环境，将获取到的框架插件包上传到安装环境任意路径（例如“/home/package”）。

    2. 执行如下命令安装TF Adapter。

        ```bash
        pip3 install "npu_device-2.6.5-py3-none-manylinux2014_<arch>.whl" -t "${TFPLUGIN_INSTALL_PATH}"
        ```

        - “npu_device-2.6.5-py3-none-manylinux2014_<arch\>.whl”请替换为安装包名称。
        - “-t”参数指定的$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包的安装路径，例如“/home/HwHiAiUser/Ascend/tfplugin”。

        pip常用命令参数如下，更多pip命令参数可执行“pip3 --help”查看。

        - --disable-pip-version-check：用于屏蔽pip版本检查，避免pip升级告警。
        - --upgrade：用于升级指定的软件包。
        - --no-deps：忽略所安装软件包的依赖。
        - --force-reinstall：强制重新安装指定的软件包。

3. 设置TF Adapter环境变量，让开发者后续可直接使用TF Adapter的Python库。

    ```bash
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
    ```

    其中$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包的安装路径。

## 升级插件包

若您需要升级TF Adapter插件包，需要先卸载，再安装。

1. 卸载TF Adapter软件包。
    - **场景1**：TF Adapter 8.0.0版本首次采用whl打包形式，若您环境中安装的是TF Adapter 8.0.0之前版本，需要参见以下步骤卸载。
        1. 执行如下命令，进入卸载脚本所在路径。

            ```bash
            cd /home/HwHiAiUser/Ascend/tfplugin/latest/script
            ```

            其中“/home/HwHiAiUser”为8.0.0之前版本的TF Adapter插件包安装路径，请根据实际情况替换。

        2. 执行uninstall.sh脚本，卸载老版本TF Adapter软件包。

            ```bash
            ./uninstall.sh
            ```

    - **场景2**：若您安装的是TF Adapter 8.0.0及之后版本的whl包，请参见下文[卸载插件包](#卸载插件包)。

2. 卸载成功后，参见[安装插件包](#安装插件包)安装新版本TF Adapter whl包。

## 卸载插件包

安装完TF Adapter插件包后，若您需要卸载，可执行如下命令。

```bash
## 设置环境变量，指定TF Adapter软件Python库所在路径
export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

# 卸载TensorFlow 2.6.5框架适配插件包
pip3 uninstall -y npu_device
```
