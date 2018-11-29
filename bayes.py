import pandas as pd 
import numpy as np 
import math as m 
from maths import mathematica
from modifyInput import modifier
class bayes:
	def __init__(self, dataLink):
		self.meanList=[]
		self.covMatrix=[]
		self.df= pd.read_csv(dataLink)
		self.num_training_set=self.df.shape[0] 
		self.num_features= self.df.shape[1]-1
		self.data = modifier.modify(self.df)

	def fit(self):		
		self.meanList = mathematica.meanClassCond(self.data, self.num_features)
		self.covMatrix= mathematica.covMatrix(self.data, self.num_features, self.meanList)
	
	def predict(self, inputVector):
		l1 = []
		for i in range(0, len(self.meanList)):
			prod=1
			l2= []; l2.append(self.meanList[i][0])
			for j in range(0, len(self.meanList[i][1])):
				prod = prod * mathematica.gaussian(self.meanList[i][1][j], self.varList[i][1][j], inputVector[j])
			l2.append(prod)
			l1.append(l2)
		maxInd= mathematica.findMax(l1)
		return l1[maxInd][0]
