import pandas as pd 
from sklearn.model_selection import train_test_split 
import numpy as np 
from sklearn import svm

df = pd.read_csv("Medical_data.csv")
tf = pd.read_csv("test_medical.csv")
x_train= df.iloc[:, 1:df.shape[1]]
y_train= df.iloc[:,0]
#print(allLabels)
x_test= tf.iloc[:, 1:tf.shape[1]]
y_test= tf.iloc[:,0]
cls = svm.SVC(C=7.0, kernel="rbf", decision_function_shape="ovo")
cls.fit(x_train,y_train)
print(cls.score(x_test, y_test))
