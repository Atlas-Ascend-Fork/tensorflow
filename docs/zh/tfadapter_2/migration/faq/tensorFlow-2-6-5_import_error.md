# 安装TensorFlow 2.6.5后，执行import tensorflow时报错

## 问题描述

安装TensorFlow 2.6.5后，执行import tensorflow时报错：“RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xd”。

**图 1**  报错截图
![报错截图](../figures/import_tensorflow_error.png)

## 可能原因

pip3安装TensorFlow的时候，可能会自动重装numpy，导致TensorFlow和numpy版本不兼容，需用户手动重装numpy。

## 解决措施

执行如下命令卸载旧版本numpy，安装TensorFlow 2.6.5适配的numpy 1.23.0版本。

```bash
pip3 uninstall numpy
pip3 install numpy==1.23.0
```
