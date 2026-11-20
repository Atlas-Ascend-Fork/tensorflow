# An Error Is Reported When import tensorflow Is Executed After TensorFlow 2.6.5 Is Installed

## Symptom

After TensorFlow 2.6.5 is installed, the error message "RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xd" is displayed when **import tensorflow** is executed.

**Figure  1**  Screenshot of the error
![](../figures/import_tensorflow_error.png)

## Possible Cause

When TensorFlow is installed using pip3, NumPy may be automatically reinstalled. As a result, the TensorFlow and NumPy versions are incompatible and you need to manually reinstall NumPy.

## Solution

Run the following command to uninstall NumPy of the earlier version and install NumPy 1.23.0 that adapts to TensorFlow 2.6.5:

```bash
pip3 uninstall numpy
pip3 install numpy==1.23.0
```
