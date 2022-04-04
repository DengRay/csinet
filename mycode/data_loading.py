import numpy as np
import h5py
import csipy.io as sio
import math
from torch.utils.data import Dataset

def dataloading(dataset_spec,envir = "indoor",img_channels = 2, img_height = 32, img_width = 32, data_format = "channels_first", T = 10,val_flag = 0):
    if envir == 'indoor':
        mat = sio.loadmat(f"{dataset_spec}/DATA_Htrainin.mat") #归一化为data2 效果好于没有归一化
        x_train = mat['HT'] # array
        mat = sio.loadmat(f"{dataset_spec}/DATA_Htestin.mat")
        x_test = mat['HT'] # array
        if val_flag:
            mat = sio.loadmat(f"{dataset_spec}/DATA_Hvalin.mat")
            x_val = mat['HT'] # array

    elif envir == 'outdoor':
        mat = sio.loadmat(f"{dataset_spec}/DATA_Htrainout.mat") 
        x_train = mat['HT'] # array
        mat = sio.loadmat(f"{dataset_spec}/DATA_Htestout.mat")
        x_test = mat['HT'] # array
        if val_flag:
            mat = sio.loadmat(f"{dataset_spec}/DATA_Hvalout.mat")
            x_val = mat['HT'] # array

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if val_flag:
        x_val = x_val.astype('float32')

    if data_format == 'channels_first':
        x_train = np.reshape(x_train, (len(x_train), T,img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
        x_test = np.reshape(x_test, (len(x_test), T,img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
        if val_flag:
            x_val = np.reshape(x_val, (len(x_val), T,img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

    elif data_format == 'channels_last':
        x_train = np.reshape(x_train, (len(x_train), T, img_height, img_width,img_channels))  # adapt this if using `channels_first` image data format
        x_test = np.reshape(x_test, (len(x_test), T, img_height, img_width,img_channels))  # adapt this if using `channels_first` image data format
        if val_flag:
            x_val = np.reshape(x_val, (len(x_val), T, img_height, img_width,img_channels))  # adapt this if using `channels_first` image data format

    if val_flag:
        return x_train,x_val,x_test
    else:
        return x_train,x_val
        
def power(x):
    x_real = np.reshape(x[:, :, 0, :, :], (x.shape[0]*x.shape[1], -1))
    x_imag = np.reshape(x[:, :, 1, :, :], (x.shape[0]*x.shape[1], -1))
    x_C = x_real + 1j*(x_imag)
    pow = np.sum(abs(x_C)**2, axis=2)
    pow = np.sqrt(pow)
    return pow

def pre_process(x):
    pow=power(x)
    for i in range (x.shape[0]):
        for j in range (x.shape[1])
        x[i, j, :]=x[i, j, :]/pow[i][j]
    return x


