# FAQ

## 无法安装swig软件包

### 问题现象

在安装TF Serving编译依赖的软件包时可能会出现无法安装swig软件包问题，并显示如下报错信息。

```text
E: The package ascend-cann-toolkit needs to be reinstalled, but I can't  find an archive for it.
```

### 解决方案

可根据以下操作步骤重新安装swig依赖包。

1. 备份“/var/lib/dpkg/status“。

    ```bash
    cp /var/lib/dpkg/status /{newfilepath}/status
    ```

    \{newfilepath\}表示需要备份的路径，请根据实际情况替换。

2. 打开“/var/lib/dpkg/status“文件，定位到出错的软件包记录，然后在此文件中将该软件包记录删除，如下图中标注位置所示。

    ```bash
    vim /var/lib/dpkg/status
    ```

    ![](../../figures/swing_error.png "出错的软件包记录")

3. 重新安装swig。

    ```bash
    apt install swig
    ```

## TF Serving编译时提示缺少builtins

### 问题现象

在TF Serving编译过程中可能会出现builtins依赖模块查询失败问题，报错如下图所示。

![](../../figures/builtins_loss_error.png "缺少builtins导致的报错信息")

### 解决方案

原因是缺失“future“依赖包，解决方案是安装“future“依赖包，并检查Python指向。

1. 执行如下命令，安装future依赖包。

    ```bash
    pip3.7 install future
    ```

2. 检查“Python“软链接是否指向Python3.7.5。

    由于TF Serving编译时需要使用3.7.5版本的Python，TF Serving的相关脚本默认使用的Python解释器关键字“Python“，与系统“Python“软链接默认指向的2.7版本Python不匹配。因此需要将当前“Python“软链接指向“Python3.7.5“。

    1. 执行如下命令，检查Python软链接是否指向了Python3.7.5。

        ```bash
        python --version
        ```

        若Python为3.7.5版本，可直接重新编译TF Serving，反之继续执行之后流程。

    2. 执行如下命令创建软链接，将“Python“指向“Python3.7.5“。

        ```bash
        ln -sf /usr/local/python3.7.5/bin/python3.7  /usr/bin/python
        ```

3. 重新编译TF Serving。
4. （可选）执行以下命令，复原“Python“软链接指向。

    ```bash
    ln -sf /usr/bin/python2 /usr/bin/python
    ```

## TF Adapter编译结束未生成tfadapter.tar

### 问题现象

提示编译成功，但output目录下未生成tfadapter.tar。

### 解决方案

解决方案为修改“tf_adapter_2.x/CI_Build“文件，并重新编译。

1. 在相应路径下执行如下命令，打开需要修改的配置文件。

    ```bash
    vim tf_adapter_2.x/CI_Build
    ```

    修改内容如下：

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

    # 原代码如下：
    if [ "$(arch)" != "x86_64" ];then
      mkdir -p build/dist/python/dist/
      touch build/dist/python/dist/npu_device-0.1-py3-none-any.whl
      exit 0
    fi
    ```

2. 重新编译TF Adapter。
