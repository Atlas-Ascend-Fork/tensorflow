# Preparations

## Original Model Preparation

ResNet-50 is a deep residual network competent for CIFAR-10 and ImageNet-1K classification tasks.

Click  [here](https://github.com/tensorflow/models/tree/r2.1.0/official)  to obtain the original ResNet script.

The directory is organized as follows. \(Only certain involved files are listed. For additional files, see the original ResNet script.\)

```text
├── r1   // Original model directory.
│   ├── resnet    // ResNet main directory.
│        ├── __init__.py
│        ├── imagenet_main.py      // Script for training the network based on the ImageNet dataset.
│        ├── imagenet_preprocessing.py     // ImageNet preprocessing module.
│        ├── resnet_model.py     // ResNet model file.
│        ├── resnet_run_loop.py    // Data input processing and execution iteration (training, validation, and test).
│        ├── README.md   // Project description file.
│   ├── utils
│        ├── export.py     // Data receive functions, which define the parameter formats that the exported model can respond to.
├── utils
│   ├── flags
│        ├── core.py         // Public APIs including the parameter definition.
│   ├── logs
│        ├── hooks_helper.py     // Tool used for custom model testing and training, such as the function for calculating the number of steps per second, and the function of capturing CPU/GPU analysis information.
│        ├── logger.py      // Log tool.
│   ├── misc
│        ├── distribution_utils.py       // Auxiliary functions used for running models in distributed mode.
│        ├── model_helpers.py      // Functions that can be called by the model, such as a function that stops the model.
```

## Dataset Preparation

1. Prepare a dataset.
    1. Obtain the dataset.

        This sample uses the ImageNet 2012 dataset as an example. Download the dataset from  [https://www.image-net.org/](https://www.image-net.org/). Validate the dataset package and upload it to the training environment.

    2. The dataset directory is organized as follows \(the dataset directory is  **/data/dataset/**  in this example\):

        ```text
        ├──imagenet2012
        │   ├──ILSVRC2012_img_train.tar
        │   ├──ILSVRC2012_img_val.tar
        │   ├──ILSVRC2012_bbox_train_v2.tar.gz
        ```

    3. Create and run the training script. Create the  **train**,  **val**,  **bbox**, and  **imagenet_tf**  directories respectively, and decompress the  **train**,  **val**, and  **bbox**  dataset packages to the corresponding directories.
        1. Create and open the  **prepare_dataset.sh**  script.

            **vim prepare_dataset.sh**

        2. Add the following commands to the script:

            ```bash
            #!/bin/bash
            mkdir -p train val bbox imagenet_tf
            tar -xvf ILSVRC2012_img_train.tar -C train/
            tar -xvf ILSVRC2012_img_val.tar -C val/
            tar -xvf ILSVRC2012_bbox_train_v2.tar.gz -C bbox/
            ```

        3. Run the  **:wq!**  command to save the file and exit.
        4. Run the following command to run the script:

            **bash prepare_dataset.sh**

            If a child .tar package is extracted from the  **train**  package, run the following command in the  **train**  directory to decompress the child package:

            `find . -name "*.tar" | while read LINE ; do mkdir -p "${LINE%.tar}"; tar -xvf "${LINE}" -C "${LINE%.tar}"; rm -f "${LINE}"; done`

    4. Check the dataset directory. In this example, the dataset directory is  **/data/dataset/**.

        ```text
        ├──imagenet2012
        │   ├──ILSVRC2012_img_train.tar
        │   ├──ILSVRC2012_img_val.tar
        │   ├──ILSVRC2012_bbox_train_v2.tar.gz
        │   ├──bbox/
        │   ├──train/
        │   ├──val/
        ```

2. Convert the dataset to the TFRecord format.
    1. Download the source package.

        **git clone** [https://github.com/tensorflow/models.git](https://github.com/tensorflow/models.git)

    2. Go to the  **datasets**  directory of the source package and preprocess validation data.

       ```bash
       cd models-master/research/slim/datasets/

       python preprocess_imagenet_validation_data.py /data/dataset/imagenet2012/val/ imagenet_2012_validation_synset_labels.txt  # Preprocess the validation data
       ```

    3. Convert XML files to a CSV file.

       ```bash
       python process_bounding_boxes.py /data/dataset/imagenet2012/bbox/ imagenet_lsvrc_2015_synsets.txt | sort > imagenet_2012_bounding_boxes.csv
       ```

    4. Convert the ImageNet dataset to the TFRecord format.

       ```bash
       python build_imagenet_data.py --output_directory=/data/dataset/imagenet2012/imagenet_tf --train_directory=/data/dataset/imagenet2012/train --validation_directory=/data/dataset/imagenet2012/val
       ```

3. Inspect the dataset after conversion.

    ```text
    ├─ imagenet2012
    ├─├─imagenet_tf
    │     ├──train-00000-of-01024
    │     ├──train-00001-of-01024
    │     ├──train-00002-of-01024
    │     ...
    │     ├──validation-00000-of-00128
    │     ├──validation-00001-of-00128
    │     ├──validation-00002-of-00128
    │     ...
    ```
