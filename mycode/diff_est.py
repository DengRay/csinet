import numpy as np
import math
import random
from scipy import optimize, special

def diff(Ht,Ht_pre):
    Ht_real = Ht[:,0,:,:]
    Ht_imag = Ht[:,1,:,:]
    Ht_C = Ht_real + 1j*Ht_imag
    Ht_pre_real = Ht_pre[:,0,:,:]
    Ht_pre_imag = Ht_pre[:,1,:,:]
    Ht_pre_C = Ht_pre_real + 1j*Ht_pre_imag
    temp1 = np.mean(np.trace(Ht_C*np.conj(Ht_pre_C),axis1=1,axis2=2))
    temp2 = np.mean(abs(Ht_pre_C)**2)
    coeff = temp1/temp2
    dif = Ht-Ht_pre*coeff
    return [coeff,dif]

def my_diff(Ht,Ht_pre):
    Ht_real = Ht[:,0,:,:]
    Ht_imag = Ht[:,1,:,:]
    Ht_C = Ht_real + 1j*Ht_imag
    Ht_pre_real = Ht_pre[:,0,:,:]
    Ht_pre_imag = Ht_pre[:,1,:,:]
    Ht_pre_C = Ht_pre_real + 1j*Ht_pre_imag
    velocity = 0.001
    wavelenth = 0.0283*2
    snap_time = 0.04
    temp1_fdts = velocity / wavelenth * random.random() * snap_time
    coeff = 2*math.pi*special.jv(0,temp1_fdts)
    dif = Ht-Ht_pre*coeff
    return [coeff,dif]



    
