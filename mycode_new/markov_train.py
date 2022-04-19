import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import concatenate,Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
import os
from csinet_markov import *
from csinet_pro import *
from data_loading import *
from diff_est import *
from nmse import *
from test_dataloading import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
T = 10
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim_hi = 512 
encoded_dim_lo = 128

dataset_spec = '/public/git/dataT'
dataset_tail = 'DATA_HT10_'
#x_train,x_val = dataloading(dataset_spec,dataset_tail,envir = "indoor",img_channels = 2, img_height = 32, img_width = 32, data_format = "channels_first", T = 10,val_flag = 0)
x_train,x_val = dataload()
coeff = my_diff()
print(np.shape(x_train))
print(np.shape(x_val))
#x_train_pre = np.transpose(x_train_pre,[1,0,2,3,4])
#print(np.shape(x_train_pre))
#print(x_train_pre)
#x_train_pre = np.squeeze(x_train_pre)
#x_val_pre = np.squeeze(x_val_pre)
#print(np.shape(x_train_pre))

x_train_p=power(x_train)
x_val_p=power(x_val)
x_train_pre=pre_process(x_train)
x_val_pre=pre_process(x_val)
'''
for i in range(T):
    if T >= 1:
        temp1 = x_train_pre[:,i-1,:,:,:] 
        temp2 = x_train_pre[:,i,:,:,:] 
        temp3 = x_val_pre[:,i-1,:,:,:]
        temp4 = x_val_pre[:,i,:,:,:]
'''

class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []
    
        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))
            
        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('val_loss'))

history = LossHistory()
file = 'CsiNet_markov_'+(envir)+'_dim'+str(encoded_dim_lo)+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' %file 
    #outpath_base = f"{model_dir}/{opt.env}"
    #if opt.dir != None:
    #    outpath_base += "/" + opt.dir 
    #outfile_base = f"{outpath_base}/cr{opt.rate}/{network_name}"
    #subnetwork_spec = [outpath_base, subnetwork_name]

autoencoder = markov_net(coeff,img_channels, img_height, img_width,T,encoded_dim_hi,encoded_dim_lo,data_format='channels_first')

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
x_hat = autoencoder.predict(x_val_pre)
tEnd = time.time()
print ("It cost %f sec" % ((tEnd - tStart)/x_val_pre.shape[0]))

calc_NMSE(x_hat,x_val_pre,T=10,pow_diff=None)
