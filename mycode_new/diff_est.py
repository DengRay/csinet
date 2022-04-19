import numpy as tf
import tensorflow.compat.v1 as tf
import math
import random
from scipy import optimize, special
from tensorflow.keras.layers import concatenate

def diff(Ht,Ht_pre):
    Ht_real = Ht[:,0,:,:]
    Ht_imag = Ht[:,1,:,:]
    #Ht_C = Ht_real + 1j*Ht_imag
    Ht_pre_real = Ht_pre[:,0,:,:]
    Ht_pre_imag = Ht_pre[:,1,:,:]
    #Ht_pre_C = Ht_pre_real + 1j*Ht_pre_imag
    temp1 = tf.reduce_mean(tf.trace(Ht_real*tf.conj(Ht_pre_real)))
    temp2 = tf.reduce_mean(tf.trace(Ht_imag*tf.conj(Ht_pre_imag)))
    temp3 = tf.reduce_mean(tf.abs(Ht_pre_real)**2)
    temp4 = tf.reduce_mean(tf.abs(Ht_pre_imag)**2)
    coeff1 = temp1/temp3
    coeff2 = temp2/temp4
    dif1 = Ht_real-Ht_pre_real*coeff1
    dif2 = Ht_imag-Ht_pre_imag*coeff2
    dif1 = tf.expand_dims(dif1, axis=1)
    dif2 = tf.expand_dims(dif2, axis=1)
    dif = concatenate([dif1,dif2],axis=1)
    return [coeff1,coeff2,dif]

def inv_diff(coeff1,coeff2,dif):
    dif1 = dif[:,0,:,:]
    dif2 = dif[:,1,:,:]
    dif1 = dif1*coeff1
    dif2 = dif2*coeff2
    dif1 = tf.expand_dims(dif1, axis=1)
    dif2 = tf.expand_dims(dif2, axis=1)
    dif_temp = concatenate([dif1,dif2],axis=1)
    return dif_temp

def ori_diff(Ht,Ht_pre):
    Ht_real = Ht[:,0,:,:]
    Ht_imag = Ht[:,1,:,:]
    Ht_C = Ht_real + 1j*Ht_imag
    Ht_pre_real = Ht_pre[:,0,:,:]
    Ht_pre_imag = Ht_pre[:,1,:,:]
    Ht_pre_C = Ht_pre_real + 1j*Ht_pre_imag
    temp1 = tf.reduce_mean(tf.trace(Ht_C*tf.conj(Ht_pre_C),axis1=1,axis2=2))
    temp2 = tf.reduce_mean(abs(Ht_pre_c)**2)
    coeff = temp1/temp2
    dif = Ht-Ht_pre*coeff
    return [coeff,dif]

def my_diff():
    '''
    Ht_real = Ht[:,0,:,:]
    Ht_imag = Ht[:,1,:,:]
    Ht_C = Ht_real + 1j*Ht_imag
    Ht_pre_real = Ht_pre[:,0,:,:]
    Ht_pre_imag = Ht_pre[:,1,:,:]
    Ht_pre_C = Ht_pre_real + 1j*Ht_pre_imag
    '''
    velocity = 0.001
    wavelenth = 0.0283*2
    snap_time = 0.04
    temp1_fdts = velocity / wavelenth * random.random() * snap_time
    coeff = 2*math.pi*special.jv(0,temp1_fdts)
    #dif = Ht-Ht_pre*coeff
    return coeff



    
