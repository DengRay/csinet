import numpy as np
import h5py
import scipy.io as sio
import math

def dataloading(dataset_spec,dataset_tail,envir = "indoor",img_channels = 2, img_height = 32, img_width = 32, data_format = "channels_first", T = 10,val_flag = 0):
    split = 0.6
    for timeslot in range(1,11):
        train_address = f"{dataset_spec}/{dataset_tail}trainin_{timeslot}.mat"
        test_address = f"{dataset_spec}/{dataset_tail}testin_{timeslot}.mat"
        '''
        mat_temp_1 = sio.loadmat(train_address)
        mat_train = mat_temp_1['HT']
        mat_temp_2 = sio.loadmat(test_address)
        mat_val = mat_temp_2['HT']
        '''
        with h5py.File(train_address, 'r') as aa:
            mat_train = aa[f"HT_{timeslot}"][()]
            aa.close()
        with h5py.File(test_address, 'r') as bb:
            mat_test = bb[f"HT_{timeslot}"][()]
            bb.close()
        ''' 
        mat_temp_1 = h5py.File(train_address)
        mat_train = mat_temp_1[f'HT_{timeslot}'][()]
        mat_temp_2 = h5py.File(test_address)
        mat_val = mat_temp_2[f'HT_{timeslot}'][()]
        '''
        mat_train = np.expand_dims(mat_train, axis=1)
        mat_test = np.expand_dims(mat_test, axis=1)
        #np.reshape(mat_val,(mat_val[0],1,2048))
        if timeslot == 1:
            x_train = mat_train
            x_test = mat_test
        else:
            x_train = np.concatenate((x_train,mat_train),axis=1)
            x_test = np.concatenate((x_test,mat_test),axis=1)


    '''   
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
    '''
    x_train = np.transpose(x_train,[2,1,0])
    x_test = np.transpose(x_test,[2,1,0])

    temp_1 = int(split * x_train.shape[0])
    mat_test = mat_test[:temp_1,:,:]
    temp_2 = int(split * x_test.shape[0])
    mat_test = mat_test[:temp_2,:,:]

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
        return x_train,x_test

def power(x):
    x_real = np.reshape(x[:, :, 0, :, :], (x.shape[0],x.shape[1], -1))
    x_imag = np.reshape(x[:, :, 1, :, :], (x.shape[0],x.shape[1], -1))
    x_C = x_real + 1j*(x_imag)
    pow = np.sum(abs(x_C)**2, axis=2)
    pow = np.sqrt(pow)
    return pow

def pre_process(x):
    pow=power(x)
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            x[i, j, :, :, :]=x[i, j, :, :, :]/pow[i][j]
    return x

if __name__ == "__main__":
    import numpy as np
    import h5py
    import scipy.io as sio
    import math
    envir = 'indoor' #'indoor' or 'outdoor'
    # image params
    img_height = 32
    img_width = 32
    img_channels = 2 
    img_total = img_height*img_width*img_channels
    # network params
    residual_num = 2
    encoded_dim_hi = 512 
    encoded_dim_lo = 64
    T = 10

    dataset_spec = '/public/git/dataT'
    dataset_tail = 'DATA_HT10_'
    x_train,x_val = dataloading(dataset_spec,dataset_tail,envir = "indoor",img_channels = 2, img_height = 32, img_width = 32, data_format = "channels_first", T = 10,val_flag = 0)
    print(np.shape(x_train))
    print(np.shape(x_val))