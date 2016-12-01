import numpy as np
import scipy.io as sio

def load_dataset():
    mat_img_train = sio.loadmat('train_img.mat')
    X_train = mat_img_train.get('J')
    X_train = X_train / np.float32(256)

    mat_label_train = sio.loadmat('train_label.mat')
    y_train = mat_label_train.get('labels')
    y_train = np.transpose(y_train)
    y_train = y_train.ravel()

    mat_img_test = sio.loadmat('test_img.mat')
    X_test = mat_img_test.get('I')
    X_test = X_test / np.float32(256)

    mat_label_test = sio.loadmat('test_label.mat')
    y_test = mat_label_test.get('labels')
    y_test = np.transpose(y_test)
    y_test = y_test.ravel()

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    return X_train, y_train, X_val, y_val, X_test, y_test
