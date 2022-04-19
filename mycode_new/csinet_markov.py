import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import concatenate,Input, Lambda,Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model
import numpy as np
from csinet_pro import *
from diff_est import *

img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels


def markov_net(coeff,img_channels, img_height, img_width,T,encoded_dim_hi,encoded_dim_lo,data_format='channels_first'):
    if(data_format == "channels_last"):
        x = Input((T, img_height, img_width, img_channels))
    elif(data_format == "channels_first"):
        x = Input((T, img_channels, img_height, img_width))
    else:
        print("Unexpected data_format param in CsiNet input.") # raise an exception eventually. For now, print a complaint
    csiori = ori_net()
    csinet_hi = csinet_pro(img_channels,img_height,img_width,encoded_dim_hi,data_format)
    csinet_lo = csinet_pro(img_channels,img_height,img_width,encoded_dim_lo,data_format)
    CsiOut = []
    for i in range(T):
        CsiIn = Lambda( lambda x:x[:,i,:,:,:])(x)
        if i == 0:
            #outlayer = csinet_hi([CsiIn])
            outlayer = csiori(CsiIn)
        else:
            temp = outlayer
            #coeff1,coeff2,dif = diff(CsiIn,temp)
            diff = CsiIn - temp
            outlayer = csinet_lo([diff])
            #outlayer = csinet_lo([CsiIn])
            #dif_temp = inv_diff(coeff1,coeff2,temp)
            outlayer = temp + outlayer
            #outlayer = dif_temp + outlayer
        if data_format == "channels_last":
            CsiOut.append(Reshape((1,img_height,img_width,img_channels))(outlayer)) 
        if data_format == "channels_first":
            CsiOut.append(Reshape((1,img_channels,img_height,img_width))(outlayer)) 

    Csiout =  concatenate(CsiOut,axis=1)
    full_model = Model(inputs=[x], outputs=[Csiout])
    full_model.compile(optimizer='adam', loss='mse')
    full_model.summary()
    return full_model
