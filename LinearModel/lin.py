import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

for deg in range(2, 10):
    
    df= pd.read_csv("Assignment2Data_4.csv")
    x_all = df.iloc[:, 0:df.shape[1]-1]
    y_all = df.iloc[:, df.shape[1]-1]
    y_all = np.asarray(y_all.values.tolist())
    x_all= np.asarray(x_all.values.tolist())
    df=df.drop(['x'], axis=1)

    for i in range(0, deg):
        ar= np.power(x_all, i)
        df.insert(i, i, ar, allow_duplicates=True)
        
    x_all = df.iloc[:, 0:df.shape[1]-1]
    y_all = df.iloc[:, df.shape[1]-1]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size= 0.05, random_state=42)
    x_train = np.asarray(x_train.values.tolist())
    y_train = np.asarray(y_train.values.tolist())
    x_test = np.asarray(x_test.values.tolist())
    y_test = np.asarray(y_test.values.tolist())
    
    a=np.matmul(x_train.T, x_train)
    b=np.linalg.inv(a)
    c=np.matmul(b, x_train.T)
    param = np.matmul(c, y_train)
    param = np.reshape(param, (deg, 1))

    y_out = np.matmul(x_test, param)
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    diff = (y_test-y_out)
    diff = np.power(diff, 2)
    err = np.sum(diff)/y_test.shape[0]
    print(err)