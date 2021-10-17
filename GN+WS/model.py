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

class pool_Conv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_shape,**kwargs):
        super(pool_Conv2D, self).__init__(**kwargs)
        self.kernel_shape = kernel_shape
        self.k_1 = tf.Variable(tf.random.truncated_normal(self.kernel_shape, mean=0.0, stddev=0.1), trainable=True)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_shape': self.kernel_shape
        })
        return config

    def call(self, inputs_tensor):
        x1 = tf.nn.conv2d(inputs_tensor, self.k_1, strides=(1,2,2,1), padding='SAME')
        self.k_1.assign(self.weight_std(self.k_1))
        x1 = tfa.activations.mish(x1)
        return x1
    
    def weight_std(self, k):
        k = (k - tf.reduce_mean(k))/(tf.math.reduce_std(k) + 1e-5)
        return k

def deconv2d(layer_input, skip_input, gt=64):
    ux = tf.keras.layers.UpSampling2D(size=2)(layer_input)
    ux = tf.keras.layers.Conv2D(gt, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(ux)
    ux = tfa.activations.mish(ux)
    ux = tf.keras.layers.Concatenate()([skip_input, ux])
    return ux

class deconv2d_nn(tf.keras.layers.Layer):
    def __init__(self, kernel_shape,**kwargs):
        super(deconv2d_nn, self).__init__(**kwargs)
        self.kernel_shape = kernel_shape
        self.k_1 = tf.Variable(tf.random.truncated_normal(self.kernel_shape, mean=0.0, stddev=0.1), trainable=True)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_shape': self.kernel_shape
        })
        return config
    def call(self, inputs_tensor,skip_input):
        w = inputs_tensor.shape[1]
        h = inputs_tensor.shape[2]
        c = inputs_tensor.shape[3]
        x = tf.nn.conv2d_transpose(inputs_tensor, self.k_1, output_shape=[1,w*2,h*2,c], strides=(1,1,1,1), padding='SAME')
        x = tf.keras.layers.Conv2D(c, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = tfa.activations.mish(x)
        x = tf.keras.layers.Concatenate()([skip_input, x])
        x = tf.keras.layers.Conv2D(c/2, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = tfa.activations.mish(x)
        return x

class build_generator(tf.keras.layers.Layer):
    def unet(gt=64,img_shape=(512,768,1)):
        # input shape
        input_layer = tf.keras.Input(shape=img_shape, name='data')
        # down C1~C6
        c1 = wsreg_Conv2D(kernel_shape=[3,3,1,gt])(input_layer) #512
        c1 = tf.keras.layers.BatchNormalization()(c1)
        d1 = pool_Conv2D(kernel_shape=[3,3,gt,gt])(c1)          #256
        d1 = tf.keras.layers.BatchNormalization()(d1)
        c2 = wsreg_Conv2D(kernel_shape=[3,3,gt,gt*2])(d1)       #256
        c2 = tf.keras.layers.BatchNormalization()(c2)
        d2 = pool_Conv2D(kernel_shape=[3,3,gt*2,gt*2])(c2)      #128
        d2 = tf.keras.layers.BatchNormalization()(d2)
        c3 = wsreg_Conv2D(kernel_shape=[3,3,gt*2,gt*4])(d2)     #128
        c3 = tf.keras.layers.BatchNormalization()(c3)
        d3 = pool_Conv2D(kernel_shape=[3,3,gt*4,gt*4])(c3)      #64
        d3 = tf.keras.layers.BatchNormalization()(d3)
        c4 = wsreg_Conv2D(kernel_shape=[3,3,gt*4,gt*8])(d3)     #64
        c4 = tf.keras.layers.BatchNormalization()(c4)
        d4 = pool_Conv2D(kernel_shape=[3,3,gt*8,gt*8])(c4)      #32
        d4 = tf.keras.layers.BatchNormalization()(d4)
        
        # UP
        u = deconv2d(d4,c4,gt*8)                                  #64
        u = deconv2d(u,c3,gt*4)                                   #128
        u = deconv2d(u,c2,gt*2)                                   #256
        u = deconv2d(u,c1,gt)                                   #512
        
        # UP2
        # u = deconv2d_nn(kernel_shape=[3,3,gt*4,gt*8])(c4,c3)
        # u = deconv2d_nn(kernel_shape=[3,3,gt*2,gt*4])(u,c2)
        # u = deconv2d_nn(kernel_shape=[3,3,gt*1,gt*2])(u,c1)

        output_img = tf.keras.layers.Conv2D(img_shape[-1], kernel_size=3, strides=1, padding='same', activation='sigmoid',kernel_initializer='he_normal')(u)
        model = tf.keras.Model(inputs=[input_layer], outputs=output_img)

        return model
  
# if __name__ == '__main__':
#     model = build_generator.unet()
#     model.summary()
#%%
class build_discriminator(tf.keras.layers.Layer):
    def down_CNN(img_shape = (512,768,1),gt = 64):

        img_A = tf.keras.Input(shape=img_shape)
        img_B = tf.keras.Input(shape=img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = tf.keras.layers.Concatenate(axis=-1)([img_A, img_B])

        x = wsreg_Conv2D(kernel_shape=[3,3,2,gt])(combined_imgs) #512
        x = tf.keras.layers.BatchNormalization()(x)
        x = pool_Conv2D(kernel_shape=[3,3,gt,gt])(x)          #256
        x = tf.keras.layers.BatchNormalization()(x)
        x = wsreg_Conv2D(kernel_shape=[3,3,gt,gt*2])(x)       #256
        x = tf.keras.layers.BatchNormalization()(x)
        x = pool_Conv2D(kernel_shape=[3,3,gt*2,gt*2])(x)      #128
        x = tf.keras.layers.BatchNormalization()(x)
        x = wsreg_Conv2D(kernel_shape=[3,3,gt*2,gt*4])(x)     #128
        x = tf.keras.layers.BatchNormalization()(x)
        x = pool_Conv2D(kernel_shape=[3,3,gt*4,gt*4])(x)      #64
        x = tf.keras.layers.BatchNormalization()(x)
        x = wsreg_Conv2D(kernel_shape=[3,3,gt*4,gt*8])(x)     #64
        x = tfa.layers.GroupNormalization()(x)

        validity = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='same')(x)

        return tf.keras.Model([img_A, img_B], validity)

# if __name__ == '__main__':
#     model = build_discriminator.down_CNN()
#     model.summary()
#%%
# model_g = build_generator.unet()
# model_d = build_discriminator.down_CNN()
def combined(generator, discriminator, img_shape = (512,768,1)):
    img_A = tf.keras.Input(shape=img_shape)
    img_B = tf.keras.Input(shape=img_shape)
    fake_A = generator(img_A)
    discriminator.trainable = False
    valid = discriminator([fake_A, img_B])
    return tf.keras.Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
# if __name__ == '__main__':
#     model = combined(model_g,model_d)
#     model.summary()