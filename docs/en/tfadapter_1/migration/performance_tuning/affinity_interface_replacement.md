# Affinity Interface Replacement

Check whether the Dropout and GELU interfaces on the original network have been replaced with the corresponding NPU interfaces. If not, replace them according to the following example.

- Replace  **dropout**  in the original network with the corresponding CANN API for better performance. You must also pay attention to the impact on the network accuracy.
  - If  **tf.nn.dropout**  exists, modify it as follows:

    ```python
    layers = npu_ops.dropout()
    ```

  - If  **tf.layers.dropout**,  **tf.layers.Dropout**,  **tf.keras.layers.Dropout**,  **tf.keras.layers.SpatialDropout1D**,  **tf.keras.layers.SpatialDropout2D**, or  **tf.keras.layers.SpatialDropout3D**  exists, add the following header file reference:

    ```python
    from npu_bridge.estimator.npu import npu_convert_dropout
    ```

- Replace  **gelu**  in the original network with the corresponding CANN API to achieve optimal performance.

    Original TensorFlow code:

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu()
    ```

    Code after porting:

    ```python
    layers = npu_unary_ops.gelu(x)
    ```
