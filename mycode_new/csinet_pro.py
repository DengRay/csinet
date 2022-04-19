import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU #Input
from tensorflow.keras import Input 
from tensorflow.keras.models import Model
import numpy as np

def ori_net():
    image_tensor = Input((2, 32, 32))
    autoencoder = Model(inputs=[image_tensor], outputs=[image_tensor])
    return autoencoder

def csinet_pro(img_channels,img_height,img_width,encoded_dim,data_format="channels_first",out_activation='tanh'):
    # build the autoencoder model of csinet_pro
    def make_csinetpro(x,encoded_dim):
        
        img_total = img_channels*img_height*img_width

        def add_common_layers(x):
            x=BatchNormalization()(x)
            x=keras.activations.tanh(x)
            return x

        x=Conv2D(16,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
        x=add_common_layers(x)
        x=Conv2D(8,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
        x=add_common_layers(x)
        x=Conv2D(4,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
        x=add_common_layers(x)
        x=Conv2D(2,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
        x=add_common_layers(x)

        x=Reshape((img_total,))(x)
        encoded=Dense(encoded_dim,activation='linear')(x)

        x=Dense(img_total,activation='linear')(encoded)
        if(data_format == "channels_first"):
            x = Reshape((img_channels, img_height, img_width,))(x)
        elif(data_format == "channels_last"):
            x = Reshape((img_height, img_width, img_channels,))(x)#x=Reshape((img_channels,img_height,img_width,))(x)
        x=Conv2D(16,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
        x=add_common_layers(x)
        x=Conv2D(8,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
        x=add_common_layers(x)
        x=Conv2D(4,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
        x=add_common_layers(x)
        x=Conv2D(2,kernel_size=(7,7),activation=out_activation,padding='same',data_format='channels_first')(x)

        return x

    if(data_format == "channels_last"):
        image_tensor = Input((img_height, img_width, img_channels))
    elif(data_format == "channels_first"):
        image_tensor = Input((img_channels, img_height, img_width))
    else:
        print("Unexpected tensor_shape param in CsiNet input.")

    network_output = make_csinetpro(image_tensor,encoded_dim)
    autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
    return autoencoder
        
