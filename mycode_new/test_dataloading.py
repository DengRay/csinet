import numpy as np
import h5py

def dataload():
    split = 0.6
    T = 10
    img_height = 32
    img_width = 32
    img_channels = 2 
    for i in range(1,11):
        train_address = f"/public/git/dataT/DATA_HT10_trainin_{i}.mat"
        with h5py.File(train_address, 'r') as bb:
            mat_train = bb[f"HT_{i}"][()]
            bb.close()
        mat_train = np.expand_dims(mat_train, axis=1)
        mat_train = np.transpose(mat_train,[2,1,0])
        temp = int(split * mat_train.shape[0])
        mat_train = mat_train[:temp,:,:]
        if i == 1:
            x_train = mat_train
        else:
            x_train = np.concatenate((x_train,mat_train),axis=1)
    x_train = x_train.astype('float32')
    x_train = np.reshape(x_train, (len(x_train), T, img_channels, img_height, img_width))
    #print(np.shape(x_train))

    for i in range(1,11):
        test_address = f"/public/git/dataT/DATA_HT10_testin_{i}.mat"
        with h5py.File(test_address, 'r') as bb:
            mat_test = bb[f"HT_{i}"][()]
            bb.close()
        mat_test = np.expand_dims(mat_test, axis=1)
        mat_test = np.transpose(mat_test,[2,1,0])
        temp = int(split * mat_test.shape[0])
        mat_test = mat_test[:temp,:,:]
        if i == 1:
            x_test = mat_test
        else:
            x_test = np.concatenate((x_test,mat_test),axis=1)
    x_test = x_test.astype('float32')
    x_test = np.reshape(x_test, (len(x_test), T, img_channels, img_height, img_width))
    #print(np.shape(x_test))
    return [x_train,x_test]


if __name__ == "__main__":
    e_train,e_test = dataload()
    print(np.shape(e_train))
    print(np.shape(e_test))
