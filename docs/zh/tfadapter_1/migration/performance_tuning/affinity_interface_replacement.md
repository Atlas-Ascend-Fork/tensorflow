# 亲和接口是否替换

对于原始网络中的dropout、gelu接口，请检查是否已替换为对应的NPU接口。若未替换，请按照如下示例进行替换。

- 对于原始网络中的dropout，请替换为CANN对应的API实现，以获得更优性能，但需关注对网络精度的影响。
  - 如果存在tf.nn.dropout，请修改为：

    ```python
    layers = npu_ops.dropout()
    ```

  - 如果存在tf.layers.dropout/tf.layers.Dropout/tf.keras.layers.Dropout/tf.keras.layers.SpatialDropout1D/tf.keras.layers.SpatialDropout2D/tf.keras.layers.SpatialDropout3D，请增加头文件引用：

    ```python
    from npu_bridge.estimator.npu import npu_convert_dropout
    ```

- 对于原始网络中的gelu，请替换为CANN对应的API实现，以获得更优性能。

    TensorFlow原始代码：

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu()
    ```

    迁移后的代码：

    ```python
    layers = npu_unary_ops.gelu(x)
    ```
