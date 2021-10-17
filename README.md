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
## GN+WS

