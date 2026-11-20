# Model Freezing

The process of saving graphs and weights using TensorFlow is called freezing. During the process, a protobuf file \(.pb file for short\) is generated. The following lists some general steps.

1. Save the model.

    ```python
    tf.saved_model.save(network, "save_path")
    ```

    The following files or folders are generated in the  **save_path**  folder:

    ```text
    |--- save_model.pb     # Network structure to be saved
    |--- variables         # Weight parameters
    |--- assets            # Directory storing required external files, for example, the initialized vocabulary file
    ```

2. Freeze the .pb model.

    Load the model exported in the preceding steps and freeze it as a .pb model with weights. The sample code is as follows:

    ```python
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from tensorflow.python.framework import graph_util

    # Load the model.
    saved_model_dir = "save_path"
    model = tf.saved_model.load(saved_model_dir)
    # Initialize signatures.
    infer = model.signatures["serving_default"]
    # Freeze the .pb file with weights.
    frozen_func = convert_variables_to_constants_v2(infer)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                         logdir="./",
                         name="frozen_graph.pb",
                         as_text=False)
    ```
