import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf.reset_default_graph()
#tf.enable_eager_execution()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 128  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

#build the autoencoder model of sphnet

def add_common_layers(x):
    #x=BatchNormalization()(x)
    x=keras.activations.tanh(x)
    return x

def sphnet(x,encoded_dim):

 #Spherical architecture(pre-process)
    '''
    x_train=x
    x_train_real=x_train[0,:,:],(1024))
    x_train_imag=np.reshape(x_train[1,:,:],(1024))
    x_train_c=x_train_real-0.5+1j*(x_train_imag-0.5)
    powere=np.sum(abs(x_train_c))
    x=x/power
    '''
    '''
    x1=x
    x1=x1[0,:,:]
    print(x1.shape)
    x2=x[:,0,:,:]
    print(x2.shape)
    print(x.shape)
    x3=x[0,:,:,:]
    print(x3.shape)
    '''
        
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
    x=Reshape((img_channels,img_height,img_width,))(x)
    x=Conv2D(16,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
    x=add_common_layers(x)
    x=Conv2D(8,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
    x=add_common_layers(x)
    x=Conv2D(4,kernel_size=(7,7),padding='same',data_format='channels_first')(x)
    x=add_common_layers(x)
    x=Conv2D(2,kernel_size=(7,7),activation='tanh',padding='same',data_format='channels_first')(x)

    #x=x*powere
    
    return x

image_tensor=Input(shape=(img_channels,img_height,img_width))
network_output=sphnet(image_tensor,encoded_dim)
autoencoder=Model(inputs=[image_tensor],outputs=[network_output])
autoencoder.compile(optimizer='adam',loss='mse')
print(autoencoder.summary())


# Data loading
if envir == 'indoor':
    mat = sio.loadmat('/rt/csi/test/data/DATA_Htrainin.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data/DATA_Hvalin.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data/DATA_Htestin.mat')
    x_test = mat['HT'] # array

elif envir == 'outdoor':
    mat = sio.loadmat('/rt/csi/test/data/DATA_Htrainout.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data/DATA_Hvalout.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data/DATA_Htestout.mat')
    x_test = mat['HT'] # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

#Spherical architecture(pre-process)

def power(x):
    x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
    x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
    x_C = x_real-0.5 + 1j*(x_imag-0.5)
    power = np.sum(abs(x_C)**2, axis=1)
    power=np.sqrt(power)
    return power

def pre_process(x):
    x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
    x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
    x_C = x_real-0.5 + 1j*(x_imag-0.5)
    power = np.sum(abs(x_C)**2, axis=1)
    power=np.sqrt(power)
    x = np.reshape(x, (len(x), -1)) 
    for i in range (x.shape[0]):
        x[i,:]=x[i,:]/power[i]
    x = np.reshape(x, (len(x), img_channels, img_height, img_width)) 
    return x

x_train_p=power(x_train)
x_val_p=power(x_val)
x_test_p=power(x_test)
x_train_pre=pre_process(x_train)
x_val_pre=pre_process(x_val)
x_test_pre=pre_process(x_test)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        

history = LossHistory()
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' %file

autoencoder.fit(x_train_pre, x_train_pre,
                epochs=100,#1000
                batch_size=1000,#200
                shuffle=True,
                #verbose=2,#没有这一行
                #use_mutiprocessing = TRUE,#没有这一行
                validation_data=(x_val_pre, x_val_pre),
                callbacks=[history,
                           TensorBoard(log_dir = path)])

filename = 'result/trainloss_%s.csv'%file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result/valloss_%s.csv'%file
loss_history = np.array(history.losses_val)
np.savetxt(filename, loss_history, delimiter=",")

#Testing data
tStart = time.time()
x_hat = autoencoder.predict(x_test_pre)
x_hat = np.reshape(x_hat, (len(x_hat), -1)) 
for i in range (x_hat.shape[0]):
    x_hat[i,:]=x_hat[i,:]*x_test_p[i]
x_hat = np.reshape(x_hat, (len(x_hat), img_channels, img_height, img_width)) 
tEnd = time.time()
print ("It cost %f sec" % ((tEnd - tStart)/x_test_pre.shape[0]))

# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('/rt/csi/test/data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']# array

elif envir == 'outdoor':
    mat = sio.loadmat('/rt/csi/test/data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']# array

X_test = np.reshape(X_test, (len(X_test), img_height, 125))
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))
rho = np.mean(aa/(n1*n2), axis=1)
X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test = np.reshape(X_test, (len(X_test), -1))
power = np.sum(abs(x_test_C)**2, axis=1)
power_d = np.sum(abs(X_hat)**2, axis=1)
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
print("Correlation is ", np.mean(rho))
filename = "result/decoded_%s.csv"%file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(filename, x_hat1, delimiter=",")
filename = "result/rho_%s.csv"%file
np.savetxt(filename, rho, delimiter=",")


import matplotlib.pyplot as plt
'''abs'''
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.show()


# save
# serialize model to JSON
model_json = autoencoder.to_json()
outfile = "result/model_%s.json"%file
with open(outfile, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
outfile = "result/model_%s.h5"%file
autoencoder.save_weights(outfile)