import pandas as pd 
import numpy as np 
import math as m 
from maths import mathematica
from modifyInput import modifier
class Bayes:
	# def __init__(self, dataframe):
	# 	self.meanList=[]
	# 	self.covMat=[]
	# 	self.varList=[]
	# 	self.df=dataframe
	# 	self.num_training_set=self.df.shape[0] 
	# 	self.num_features= self.df.shape[1]-1
	# 	#print(self.num_features)
	# 	self.data = modifier.modify(self.df)
	# 	self.prior=[]
	# 	for i in range(0, len(self.data)):
	# 		self.prior.append([self.data[i][0], len(self.data[i][1])/self.num_training_set])

	def __init__(self, dataLink):
		self.meanList=[]
		self.covMat=[]
		self.varList=[]
		self.df= pd.read_csv(dataLink)
		self.num_training_set=self.df.shape[0] 
		self.num_features= self.df.shape[1]-1
		self.data = modifier.modify(self.df)
		self.prior=[]
		for i in range(0, len(self.data)):
			self.prior.append([self.data[i][0], len(self.data[i][1])/self.num_training_set])


	def fit(self):		
		self.meanList = mathematica.meanClassCond(self.data, self.num_features)
		self.covMat  = mathematica.covMatrix(self.data, self.num_features, self.meanList)
		self.varList  = mathematica.varClassCond(self.data, self.num_features, self.meanList)

	def predict(self, inputVector):
		l1 = []
		for i in range(0, len(self.data)):
			l2=[]
			l2.append(self.data[i][0])
			l2.append(self.prior[i][1]*mathematica.multivariate(self.meanList[i][1], self.covMat[i][1],np.asarray(inputVector)))
			l1.append(l2)
		maxInd= mathematica.findMax(l1)
		return l1[maxInd][0]


    	