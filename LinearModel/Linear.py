import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

def training(degree, df):
    for i in range(0, degree):
        ar= np.power(np.asarray(x.values.tolist()), i)
        df.insert(i, i, ar, allow_duplicates=True)
    x_all = df.iloc[:, 0:df.shape[1]-1]
    y_all = df.iloc[:, df.shape[1]-1]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size= 0.001, random_state=2)
    x_train = np.asarray(x_train.values.tolist())
    y_train = np.asarray(y_train.values.tolist())
    x_test = np.asarray(x_test.values.tolist())
    y_test = np.asarray(y_test.values.tolist())
    a=np.matmul(x_train.T, x_train)
    b=np.linalg.inv(a)
    c=np.matmul(b, x_train.T)
    param = np.matmul(c, y_train)
    return param, x_train, y_train

def predict(param, input):
    return np.matmul(param.T, input)


def error(y_out, y_test):
    diff = y_out-y_test
    diff= np.power(diff, 2)
    loss = (np.sum(diff))/len(diff.tolist())
    return loss
