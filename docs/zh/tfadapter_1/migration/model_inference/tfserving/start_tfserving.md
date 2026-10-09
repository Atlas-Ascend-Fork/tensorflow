# 启动TF Serving

本节介绍启动TF Serving的操作方法，实际操作时请根据实际路径进行替换。本文中举例路径均需要确保安装用户具有读或读写权限。

1. 在任意目录下新建“tf_serving_test”文件夹（本节以$HOME目录为例），并在文件夹中创建配置文件config.cfg并添加如下内容。具体字段可参考[session配置](../../../apiref/session_config/README.md)。

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

2. （可选）加载多个模型时，在“tf_serving_test”文件夹中创建模型导入配置文件models.config并添加如下内容。

    此处以inception_v3_flowers、inception_v4和inception_v4_imagenet三个模型为例，请根据实际情况自行替换。

    ```text
    model_config_list:{
            config:{
              name:"inception_v3_flowers",      # 模型名称
              base_path:"$HOME/tf_serving_test/inception_v3_flowers",  # 模型所在路径
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

3. 在“tf_serving_test”路径下放置训练好的**SavedModel模型**，参见如下目录。

    ```text
    squeezenext/ 
    └── 1
         ├── saved_model.pb
         └── variables
             ├── variables.data-00000-of-00001
             └── variables.index
    ```

    1为版本号，请参见以上目录结构存放。

    > [!NOTE]说明
    > 如需提升TF Serving部署性能，可将SavedModel格式的模型转换为.om格式的模型，详情请参见[SavedModel模型转换om模型](common_operation.md#SavedModel模型转换om模型)。使用.om格式的模型进行在线推理时，不支持精度比对中Data Dump功能。

4. 配置环境变量
    1. 将“npu_bridge”路径添加至“LD_LIBRARY_PATH”环境变量。

        ```bash
        export LD_LIBRARY_PATH=${TFPLUGIN_INSTALL_PATH}/npu_bridge:$LD_LIBRARY_PATH
        ```

        其中$\{TFPLUGIN_INSTALL_PATH\}为TF Adapter软件包安装路径。

    2. 将“tf_adapter”路径添加至“LD_LIBRARY_PATH”环境变量。

        ```bash
        export LD_LIBRARY_PATH=$HOME/serving-1.15.0/third_party/tf_adapter:$LD_LIBRARY_PATH
        ```

    3. 请结合选用的CANN软件包，设置环境变量。

        ```bash
        # 配置CANN软件环境变量，以root用户默认安装路径为例
        source /usr/local/Ascend/cann/set_env.sh
        
        # TF Adapter python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径
        export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
        
        export JOB_ID=10087
        ```

5. 启动tensorflow_model_server，传入步骤1和步骤2中的配置文件，例如：

    单个模型时执行如下命令：

    ```bash
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_base_path=$HOME/tf_serving_test/squeezenext --model_name=squeezenext --platform_config_file=$HOME/tf_serving_test/config.cfg
    ```

    多个模型时执行如下命令：

    ```bash
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=$HOME/tf_serving_test/models.config --allow_version_labels_for_unavailable_models=true --model_config_file_poll_wait_seconds=60 --platform_config_file=$HOME/tf_serving_test/config.cfg
    ```

    > [!NOTE]说明
    > 如果重新安装其他版本CANN软件后，直接启动tensorflow_model_server服务失败，可参考[重新编译TF Serving](common_operation.md#重新编译TF-Serving)解决。

    此处需要使用绝对路径。启动成功如下图所示：

    ![](../../figures/tfserving_success.png)

    通过**tensorflow_model_server --help**命令可查看启动方式及参数，参数解释如下表所示。

    | 参数 | 说明 |
    |------|------|
    | --port | 使用gRPC方式进行通信。 |
    | --rest_api_port | 使用HTTP/REST API方式进行通信，如果设置为0则不生效，且指定的端口号必须与gRPC不同。 |
    | --model_config_file | 加载多个模型时，则需要配置此参数文件以导入多个模型，且与模型和--platform_config_file参数配置文件在同一目录下。 |
    | --model_config_file_poll_wait_seconds | 此参数设置对--model_config_file配置文件刷新时间间隔。当服务开启时，将实时刷新写入--model_config_file配置文件的模型并加载到服务端中。<br>单位“s”。 |
    | --model_name | 加载一个模型时使用该参数，其值为模型所在版本目录的父目录名。 |
    | --model_base_path | 加载的模型所在路径，若已配置--model_config_file则可忽略。 |
    | --platform_config_file | 特性配置文件。 |
