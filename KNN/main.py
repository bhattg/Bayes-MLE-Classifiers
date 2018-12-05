import pandas as pd 
import numpy as np 
from KNearestNeighbour import KNN

df = pd.read_csv("Medical_data.csv")
classifier = KNN(df, "Euclidean", 11)
testframe = pd.read_csv("test_medical.csv")
c2 = KNN(df, "Hamiltonian", 11)
c3= KNN(df, "Chebyshev", 11)
accurate = 0
testcases=testframe.shape[0]
num_features = testframe.shape[1]-1
for i in range(0,75):
	if testframe.iat[i, 0] == classifier.predict(testframe.iloc[i,1:num_features+1]):
		accurate=accurate+1
		#print(100*accurate/(i+1))
print("accuracy with Euclidean")
print(100*accurate/(75))
accurate=0
for i in range(0,75):
	if testframe.iat[i, 0] == c2.predict(testframe.iloc[i,1:num_features+1]):
		accurate=accurate+1
		#print(100*accurate/(i+1))

print("accuracy with Hamiltonian")
print(100*accurate/(75))

accurate=0
for i in range(0,75):
	if testframe.iat[i, 0] == c3.predict(testframe.iloc[i,1:num_features+1]):
		accurate=accurate+1
		#print(100*accurate/(i+1))

print("accuracy with Chebyshev")
print(100*accurate/(75))

