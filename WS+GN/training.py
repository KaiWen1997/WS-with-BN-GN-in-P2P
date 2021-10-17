#%%
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
#%%
import os
import cv2
import glob
import scipy
import scipy.io
import numpy as np
from tqdm import tqdm
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.transform import resize
from skimage.color import rgb2gray,rgba2rgb
from skimage.io import imread,imsave,imshow
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.keras.losses import binary_crossentropy,mean_squared_error

def normalize(image,smooth=1e-5):
    images = image.copy()
    img = (images[:,:]-image.min()+smooth)/(image.max()-image.min()+smooth)
    return img

def padding1(image,h=512,w=768): ## 16的倍數
    x,y = image.shape
    image = np.reshape(image,(x,y,1))
    image = normalize(image)
    pad = np.zeros((h,w,1))
    pad[0:x,0:y,:]=image[:,:,:]
    return pad
# In[] John data input
import os
import numpy as np
from sklearn.utils import shuffle
from scipy.io import loadmat, savemat

def train_data_path(train_path):
    file_path_total = []
    for fold_name in os.listdir(train_path):
        # print(fold_name)
        fold_path = train_path + '/' + fold_name
        for fatty_name in os.listdir(fold_path):
            # print(fatty_name)
            case_path = fold_path + '/' + fatty_name
            for case_name in os.listdir(case_path):
                # print(case_name)
                file_path = case_path + '/' + case_name
                for file_name in os.listdir(file_path):
                    # print(file_name)
                    file_path_total.append(file_path + '/' + file_name)
    return file_path_total

def read_data(file_path_total, h=512, w=768):
    out_B, out_E = [], []
    for file in file_path_total:
        # print(file)
        file_mat = loadmat(file)
        B_img, E_img = file_mat['B_image'], file_mat['E_image']
        
        B_img = padding1(B_img)
        E_img = padding1(E_img)
        
        out_B.append(B_img)
        out_E.append(E_img)
        
    out_B = np.stack(out_B)
    out_E = np.stack(out_E)
    return out_B, out_E

def data_gen_fn(img_name, batch_size):
    Img = shuffle(img_name)
    i=0
    while True:
        start = i * batch_size
        if (start + batch_size) > len(Img):
            end = len(Img)
            start2 = 0
            end2 = start + batch_size - len(Img)
            Ims = Img[start:end] + Img[start2:end2]
            yield read_data(Ims)
        else:
            end = start + batch_size
            yield read_data(Img[start:end])
        i=i+1
        if (i * batch_size) >= len(Img):
            Img = shuffle(img_name)
            i=0

# In[] Main
path = 'D:/Liver/Data_Set/Combine_tag/4.K_fold/Train/Model_1/'
print('------------------------------------------------------')
print(path)
batch_size  =  1
image_shape = (512,768)
file_path_total = train_data_path(path)
data_gen = data_gen_fn(file_path_total,batch_size)


steps_per_epochs = int(np.ceil(len(file_path_total) / batch_size))
epochs = 100
model_g = build_generator.unet()
model_d = build_discriminator.down_CNN()
# model_d.summary()
#%%
import tensorflow as tf
def SSIM(g, p):
    scorce = tf.image.ssim(g,p,max_val=1.0)[0]
    return 1 - scorce
# In[]
d_end = model_d.output.shape.as_list()
combine = combined(model_g,model_d)
# optimizer_d = Adam(lr = 1e-4, beta_1=0.5, epsilon=1e-8)
optimizer_g = Adam(lr = 5e-4, beta_1=0.5, epsilon=1e-8)
model_d.compile(loss=SSIM,optimizer=optimizer_g,metrics=['accuracy','mse','mae'])
combine.compile(loss=[SSIM, SSIM],loss_weights=[1, 100],optimizer=optimizer_g)
for epoch in range(epochs):
    valid = np.ones((batch_size,) + tuple(d_end[1:]))
    fake  = np.zeros((batch_size,) + tuple(d_end[1:]))
    for steps_per_epoch in tqdm(range(steps_per_epochs)):
        '''
        Train Discriminator
        '''
        # pdb.set_trace()
        imgs_A,imgs_B = next(data_gen)
        fake_A = model_g.predict(imgs_A)
        for D_times in range(10):
            d_loss_real = model_d.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = model_d.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        '''
        Train Generator
        '''
        g_loss = combine.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])
    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc:%3d%%, mse: %3d%%, mae:%3d%% ] [G loss: %f] " % (epoch+1, epochs,
                                                        steps_per_epoch+1, steps_per_epochs,
                                                        d_loss[0], 100*d_loss[1],100*d_loss[2],100*d_loss[3],
                                                        g_loss[0]))
    g_tf = 'D:/Liver/test/generator_' + str(epoch) +'_wsgn/'
    d_tf = 'D:/Liver/test/discriminator_' + str(epoch) +'_wsgn/'
    if not os.path.isdir(g_tf):
        os.mkdir(g_tf)
    if not os.path.isdir(d_tf):
        os.mkdir(d_tf)
    model_g.save_weights(g_tf)
    model_d.save_weights(d_tf)

