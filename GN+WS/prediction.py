from model import *

import os
import cv2
import glob
import scipy
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from natsort import natsorted
import tensorflow.keras as ks
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.transform import resize
from scipy.io import loadmat, savemat
from skimage.color import rgb2gray,rgba2rgb
from skimage.io import imread,imsave,imshow
from tensorflow.keras.optimizers import Adam
from skimage.measure import compare_ssim as ssim
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.keras.losses import binary_crossentropy,mean_squared_error

'''
function
'''
def normalize(image,smooth=1e-5):
    images = image.copy()
    img = (images[:,:]-image.min()+smooth)/(image.max()-image.min()+smooth)
    return img

def padding1(image,h=512,w=768): ## 16的倍數
    x,y = image.shape
    # print(x,y)
    image = np.reshape(image,(x,y,1))
    # print(image.shape)
    image = normalize(image)
    pad = np.zeros((h,w,1))
    pad[0:x,0:y,:]=image[:,:,:]
    return pad

def test_data_path(path_test_path):
    file_path_total = []
    fatty_name_list = []
    case_name_list = []
    file_name_list = []
    for fatty_name in os.listdir(path_test_path):
        # print(fatty_name)
        fatty_name_list.append(fatty_name)
        case_path = path_test_path + '/' + fatty_name
        for case_name in os.listdir(case_path):
            # print(case_name)
            case_name_list.append(case_name)
            file_path = case_path + '/' + case_name
            for file_name in os.listdir(file_path):
                # print(file_name)
                file_path_total.append(file_path + '/' + file_name)
                file_name_list.append(file_name)
    return file_path_total, fatty_name_list, case_name_list, file_name_list

def path_name_split(path_name):
    fatty = path_name.split("\\")[6]
    patient = path_name.split("\\")[7]
    file_name = path_name.split("\\")[8]
    return fatty, patient, file_name

def make3dto4d(img):
    x,y = img.shape
    image = np.zeros((1,x,y,1))
    image[0,:,:,0]=img[:]
    return image
'''
Prediction
'''
if __name__ == '__main__':
    model_g = build_generator.unet()
    model_g.load_weights("D:/Liver/test/generator_99_wsgn/")
    file_path_total, fatty_name_list, case_name_list, file_name_list = test_data_path('D:/Liver/Data_Set/Combine_tag/4.K_fold/Test/')
    
    save_path = 'D:/Liver/Predict/P2P_GAN/Pred'
    csv_col = ['image_ID','tf2_ssim','ski_ssim','tf2_psnr']
    csv_data = []
    for i in range(len(file_name_list)):
        file_mat = loadmat(file_path_total[i])
        # print(file_path_total[i])
        
        Mask = file_mat['Mask']
        Mask = Mask>0
        E_img = file_mat['E_image']
        B_img_ori = file_mat['B_image']
    
        out_B = []
        B_img = file_mat['B_image']
        B_img = padding1(B_img)
        # print(B_img.shape)
        
        out_B.append(B_img)
        out_B = np.stack(out_B)
        # print(out_B.shape)
        
        result_B = model_g.predict(out_B)
        result_B = np.squeeze(result_B)
        result_B = result_B[0:475,0:735]
        
        ssim_value1 = tf.image.ssim(tf.convert_to_tensor(make3dto4d(result_B)), tf.convert_to_tensor(make3dto4d(normalize(E_img))),max_val=1.0)[0].numpy()
        ssim_value2 = ssim(result_B, normalize(E_img))
        psnr_value = tf.image.psnr(tf.convert_to_tensor(make3dto4d(result_B)), tf.convert_to_tensor(make3dto4d(normalize(E_img))),max_val=1.0)[0].numpy()
        print('tf_SSIM:{}'.format(ssim_value1))
        print('skimage_SSIM:{}'.format(ssim_value1))
        print('tf_psnr:{}'.format(psnr_value))
        save_pos = file_path_total[i].split("/")[-1].split(".")[0]
        print(save_pos)
        csv_data.append([save_pos,ssim_value1,ssim_value2,psnr_value])
    pd.DataFrame(columns=csv_col,data=csv_data).to_csv('D:/Liver/GN+WS.csv',index=False,encoding='gbk')