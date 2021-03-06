import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

def training(degree, df, size_test, x_input):
    for i in range(0, degree):
        ar= np.power(np.asarray(x_input.values.tolist()), i)
        df.insert(i, i, ar, allow_duplicates=True)
    x_all = df.iloc[:, 0:df.shape[1]-1]
    y_all = df.iloc[:, df.shape[1]-1]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size= size_test, random_state=2)
    x_train = np.asarray(x_train.values.tolist())
    y_train = np.asarray(y_train.values.tolist())
    x_test = np.asarray(x_test.values.tolist())
    y_test = np.asarray(y_test.values.tolist())
    a=np.matmul(x_train.T, x_train)
    b=np.linalg.inv(a)
    c=np.matmul(b, x_train.T)
    param = np.matmul(c, y_train)
    return param, x_test, y_test, df

def predict(param, input, deg):
    param = np.reshape(param, (deg, 1))
    y_out = np.matmul(x_test, param)
    return y_out

def error(y_out, y_test):
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    diff = (y_test-y_out)
    diff = np.power(diff, 2)
    err = np.sum(diff)/y_test.shape[0]
    return err
