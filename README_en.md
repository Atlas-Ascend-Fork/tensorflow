# TensorFlow Adapter For Huawei Ascend

## Overview

TensorFlow Adapter For Huawei Ascend (TF Adapter) is a TensorFlow framework adaptation plugin provided by Ascend. It enables TensorFlow developers to leverage the computing power of NPUs.

Install the TF Adapter plugin and add a few configuration options to your existing TensorFlow scripts to accelerate your training tasks on Ascend AI processors.

![tfadapter](docs/zh/figures/tfadapter_overview.png)

> [!NOTE]
> TensorFlow is a trademark of Google.

## Supported TensorFlow Versions

TF Adapter supports TensorFlow 1.15 and TensorFlow 2.6.5.

## Supported Product Models

- Ascend 950PR&950DT products
- Atlas A3 products
- Atlas A2 training products
- Atlas training products
- Atlas inference products (only the TensorFlow 1.15 online inference feature is supported)

## How to Use the Source Code

If your TensorFlow framework version is 1.15, refer to [tf_adapter 1.x](./tf_adapter/README.md) for detailed compilation, installation, and usage instructions.

If your TensorFlow framework version is 2.6.5, refer to [tf_adapter 2.x](./tf_adapter_2.x/README.md) for detailed compilation, installation, and usage instructions.

## Tutorials

TF Adapter provides model migration guides, API references, training videos, and other reference materials. For details, refer to the [TF Adapter Documentation Bookshelf](./docs/README_en.md).

## Related Information

- [Contributing Guidelines](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
