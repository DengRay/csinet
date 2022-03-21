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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

def add_common_layers(x):
    x=BatchNormalization()(x)
    x=keras.activations.tanh(x)
    return x

# Bulid the autoencoder model of CsiNet
def csinetpro(x,encoded_dim):

 #Spherical architecture(pre-process)
    
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

    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = csinetpro(image_tensor, encoded_dim)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

## Data loading
if envir == 'indoor':
    mat = sio.loadmat('/rt/csi/test/data2/DATA_HTtrainin.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data2/DATA_HTvalin.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data2/DATA_HTtestin.mat')
    x_test = mat['HT'] # array

elif envir == 'outdoor':
    mat = sio.loadmat('/rt/csi/test/data1/DATA_Htrainout.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data1/DATA_Hvalout.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('/rt/csi/test/data1/DATA_Htestout.mat')
    x_test = mat['HT'] # array


x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

def power(x):
    x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
    x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
    x_C = x_real + 1j*(x_imag)
    powerr = np.sum(abs(x_C)**2, axis=1)
    powerr=np.sqrt(powerr)
    return powerr

def pre_process(x):
    powert=power(x)
    for i in range (x.shape[0]):
        x[i,:]=x[i,:]/powert[i]
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
                epochs=500,#1000
                batch_size=200,#200
                shuffle=True,
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
x_test_real = np.reshape(x_test_pre[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test_pre[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real + 1j*(x_test_imag)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real + 1j*(x_hat_imag)

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
power_q = np.sum(abs(x_test_C)**2, axis=1)
power_d = np.sum(abs(X_hat)**2, axis=1)
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power_q)))
print("Correlation is ", np.mean(rho))
filename = "result/decoded_%s.csv"%file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(filename, x_hat1, delimiter=",")
filename = "result/rho_%s.csv"%file
np.savetxt(filename, rho, delimiter=",")
