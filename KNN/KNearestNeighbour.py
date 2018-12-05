import numpy as np 
import pandas as pd 
import math as m 
from operator import itemgetter

class KNN:
	def __init__(self, df, metric, KhyperParameter):
		self.metric=metric
		self.df = df
		self.num_training_set=df.shape[0] 
		self.num_features= df.shape[1]-1
		self.KhyperParameter = KhyperParameter

	def distance(self, x1, x2):
		size  = len(x1)
		if self.metric=="Euclidean":
			return np.sqrt(np.sum(np.square(np.asarray(x1)-np.asarray(x2))))
		elif self.metric=="Hamiltonian":
			return (np.sum(np.absolute(np.asarray(x1)-np.asarray(x2))))
		elif self.metric=="Chebyshev":
			return np.amax(np.absolute(np.asarray(x1)-np.asarray(x2)))

	def predict(self, input):
		distList=[]
		class_mapping={}	
		for i in range(0, self.num_training_set):
			key=self.df.iat[i, 0]
			d=self.distance(input, (self.df.iloc[i,1:self.num_features+1]).tolist())
			distList.append((key, d))
			if key in class_mapping.keys():
				continue
			else :
				class_mapping[key]=0

		# now we have a list of tuple, with distances and class
		distList = sorted(distList, key = lambda x: x[1])
		for i in range(0, self.KhyperParameter):
			class_mapping[distList[i][0]] = class_mapping[distList[i][0]] + 1
		max=0
		predicted = ""
		for keys in class_mapping.keys():
			if class_mapping[keys]>max:
				max = class_mapping[keys]
				predicted= keys

		return predicted 
