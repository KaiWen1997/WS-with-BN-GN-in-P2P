# WS-with-BN-GN-in-P2P
使用前請先安裝`tensorflow_addons`
```
! pip install tensorflow-addons
```
用tensorflow2實現weight standardization,為了訓練方便將其建立成keras layer。 
``` python
import tensorflow_addons as tfa
import tensorflow as tf

class wsreg_Conv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_shape,**kwargs):
        super(wsreg_Conv2D, self).__init__(**kwargs)
        self.kernel_shape = kernel_shape
        self.k_1 = tf.Variable(tf.random.truncated_normal(self.kernel_shape, mean=0.0, stddev=0.1), trainable=True)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_shape': self.kernel_shape
        })
        return config

    def call(self, inputs_tensor):
        x1 = tf.nn.conv2d(inputs_tensor, self.k_1, strides=(1,1,1,1), padding='SAME')
        self.k_1.assign(self.weight_std(self.k_1))
        x1 = tfa.activations.mish(x1)
        return x1

    def weight_std(self, k):
        k = (k - tf.reduce_mean(k))/(tf.math.reduce_std(k) + 1e-5)
        return k
```
或是(來源：https://stackoverflow.com/questions/66305623/group-normalization-and-weight-standardization-in-keras)
``` python
class WSConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(kernel_initializer="he_normal", *args, **kwargs)

    def standardize_weight(self, weight, eps):

        mean = tf.math.reduce_mean(weight, axis=(0, 1, 2), keepdims=True)
        var = tf.math.reduce_variance(weight, axis=(0, 1, 2), keepdims=True)
        fan_in = np.prod(weight.shape[:-1])
        gain = self.add_weight(
            name="gain",
            shape=(weight.shape[-1],),
            initializer="ones",
            trainable=True,
            dtype=self.dtype,
        )
        scale = (
            tf.math.rsqrt(
                tf.math.maximum(var * fan_in, tf.convert_to_tensor(eps, dtype=self.dtype))
            )
            * gain
        )
        return weight * scale - (mean * scale)

    def call(self, inputs, eps=1e-4):
        self.kernel.assign(self.standardize_weight(self.kernel, eps))
        return super().call(inputs)
```

