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
'''
preprocessing
'''
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
'''
John data input
'''
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
'''
loss function
'''
import tensorflow as tf
def SSIM(g, p):
    scorce = tf.image.ssim(g,p,max_val=1.0)[0]
    return 1 - scorce

if __name__ == '__main__':
    '''
    config
    '''
    path = 'D:/Liver/Data_Set/Combine_tag/4.K_fold/Train/Model_1/'
    print('------------------------------------------------------')
    print(path)
    epochs = 100
    batch_size  =  1
    file_path_total = train_data_path(path)
    data_gen = data_gen_fn(file_path_total,batch_size)
    steps_per_epochs = int(np.ceil(len(file_path_total) / batch_size))
    '''
    models
    '''
    from model import *
    model_g = build_generator.unet()
    model_d = build_discriminator.down_CNN()
    d_end = model_d.output.shape.as_list()
    combine = combined(model_g,model_d)
    # optimizer_d = Adam(lr = 1e-4, beta_1=0.5, epsilon=1e-8)
    optimizer_g = Adam(lr = 5e-4, beta_1=0.5, epsilon=1e-8)
    model_d.compile(loss=SSIM,optimizer=optimizer_g,metrics=['accuracy','mse','mae'])
    combine.compile(loss=[SSIM, SSIM],loss_weights=[1, 100],optimizer=optimizer_g)
    '''
    training
    '''
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

