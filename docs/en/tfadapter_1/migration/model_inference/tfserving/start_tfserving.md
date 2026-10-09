# Starting TF Serving

This section describes how to start TF Serving. Replace the example paths with your actual paths as needed. Ensure that the installation user has the read, or read and write permissions on the paths described in this document.

1. Create the  **tf_serving_test**  folder in any directory \(the  **$HOME**  directory is used as an example in this section\), create the  **config.cfg**  configuration file in the folder, and add the following content to the file: For details about the fields, see  [Session Configuration](../../../apiref/session_config/README.md).

    ```text
    platform_configs {
    key: "tensorflow"
      value {
        source_adapter_config {
          [type.googleapis.com/tensorflow.serving.SavedModelBundleSourceAdapterConfig] {
            legacy_config {
              session_config {
                graph_options {
                  rewrite_options {
                    custom_optimizers {
                      name: "NpuOptimizer"
                      parameter_map: {
                        key: "use_off_line"
                        value: {
                          b: true
                        }
                      }
                      parameter_map: {
                        key: "mix_compile_mode"
                        value: {
                          b: true
                        }
                      }
                      parameter_map: {
                        key: "graph_run_mode"
                        value: {
                          i: 0
                        }
                      }
                      parameter_map: {
                        key: "precision_mode"
                        value: {
                          s: "force_fp16"
                        }
                      }                                    
                    }
                    remapping: OFF
                  }
                }
              }
            } 
          }
        }
      }
    }
    ```

2. (Optional) If multiple models are loaded, create the model import configuration file  **models.config**  in the  **tf_serving_test**  folder and add the following content.

    The  **inception_v3_flowers**,  **inception_v4**, and  **inception_v4_imagenet**  models are used as examples. Replace them with the actual model names.

    ```text
    model_config_list:{
            config:{
              name:"inception_v3_flowers",      # Model name
              base_path:"$HOME/tf_serving_test/inception_v3_flowers",  # Model path
              model_platform:"tensorflow"
     },
     config:{
              name:"inception_v4",
              base_path:"$HOME/tf_serving_test/inception_v4",
              model_platform:"tensorflow"
     },
            config:{
              name:"inception_v4_imagenet",
              base_path:"$HOME/tf_serving_test/inception_v4_imagenet",
              model_platform:"tensorflow"
            }
    }
    ```

3. Place the trained SavedModel in the  **tf_serving_test**  directory. For details, see the following directory structure.

    ```text
    squeezenext/ 
    └── 1
         ├── saved_model.pb
         └── variables
             ├── variables.data-00000-of-00001
             └── variables.index
    ```

    Wherein,  **1**  indicates the version number.

    > [!NOTE]NOTE
    >To improve the TF Serving deployment performance, you can convert the model from the SavedModel format to the .om format. For details, see  [Converting a SavedModel to an .om Model](common_operation.md#converting-a-savedmodel-to-an-om-model). When an .om model is used for online inference, the data dump function for accuracy comparison is not supported.

4. Set environment variables.
    1. Add the  **npu_bridge**  path to the environment variable  **LD_LIBRARY_PATH**.

        ```bash
        export LD_LIBRARY_PATH=${TFPLUGIN_INSTALL_PATH}/npu_bridge:$LD_LIBRARY_PATH
        ```

        $\{TFPLUGIN_INSTALL_PATH\}  is the installation path of the TF Adapter package.

    2. Add the  **tf_adapter**  path to the  **LD_LIBRARY_PATH**  environment variable.

        ```bash
        export LD_LIBRARY_PATH=$HOME/serving-1.15.0/third_party/tf_adapter:$LD_LIBRARY_PATH
        ```

    3. Set environment variables based on the selected CANN package.

        ```bash
        # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
        source /usr/local/Ascend/cann/set_env.sh
        
        # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
        export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
        
        export JOB_ID=10087
        ```

5. Start tensorflow_model_server, and import the configuration file in step 1 and step 2. For example:

    For a single model, run the following command:

    ```bash
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_base_path=$HOME/tf_serving_test/squeezenext --model_name=squeezenext --platform_config_file=$HOME/tf_serving_test/config.cfg
    ```

    For multiple models, run the following command:

    ```bash
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=$HOME/tf_serving_test/models.config --allow_version_labels_for_unavailable_models=true --model_config_file_poll_wait_seconds=60 --platform_config_file=$HOME/tf_serving_test/config.cfg
    ```

    > [!NOTE]NOTE
    >If  **tensorflow_model_server**  fails to be started after the CANN software of another version is installed, rectify the fault by referring to  [Rebuilding TF Serving](common_operation.md#rebuilding-tf-serving).

    Use an absolute path. If the startup is successful, the following information is displayed:

    ![](../../figures/tfserving_success.png)

    You can run the  **tensorflow_model_server --help**  command to view the startup mode and options. The following table describes the options.

    | Option | Description |
    | --- | --- |
    | --port | Uses the gRPC mode for communication. |
    | --rest_api_port | Uses the HTTP/REST API mode for communication. If set to 0, this option does not take effect. In addition, the specified port number must be different from that of the gRPC mode. |
    | --model_config_file | Imports multiple models. The file must be in the same directory as the models and --platform_config_file configuration file. |
    | --model_config_file_poll_wait_seconds | Sets the interval for updating the --model_config_file configuration file. When the service is enabled, the models written to --model_config_file are updated in real time and loaded to the server.<br>Unit: s. |
    | --model_name | Loads a single model. The value is the parent directory name of the version directory where the model is located. |
    | --model_base_path | Sets the path of the loaded model. If --model_config_file has been configured, ignore this option. |
    | --platform_config_file | Sets the feature configuration file. |
