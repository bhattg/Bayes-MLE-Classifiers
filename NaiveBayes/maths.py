import numpy as np
import math as m
class mathematica:

	def gaussian(mean, variance, x):
		return (1/m.sqrt(2*m.pi*variance))*m.exp((-1/(2*variance))*(x-mean)**2)

	def findMax(inputList):
		max=0;ind=0
		for i in range(0, len(inputList)):
			if max<= inputList[i][1]:
				max=inputList[i][1]
				ind=i
		return ind

	def meanClassCond(mod_training_data, num_features):
		l2=[]
		for i in range(0, len(mod_training_data)):
		    l1=[]
		    l1.append(mod_training_data[i][0])
		    sum = np.zeros(num_features)
		    for j in range(0, len(mod_training_data[i][1])):
		        sum =sum+np.asarray(mod_training_data[i][1][j])
		    sum =sum/len(mod_training_data[i][1])
		    l1.append(sum)
		    l2.append(l1)

		return l2

	def varClassCond(mod_training_data, num_features, meanList):
		l2=[]
		for i in range(0, len(mod_training_data)):
		    l1=[]
		    l1.append(mod_training_data[i][0])
		    sum = np.zeros(num_features)
		    for j in range(0, len(mod_training_data[i][1])):
		    	diff = np.asarray(mod_training_data[i][1][j]) - meanList[i][1]
		    	diff = np.square(diff)
		    	sum = sum+ diff
		    sum =sum/len(mod_training_data[i][1])
		    l1.append(sum)
		    l2.append(l1)
		return l2

	def cov(perClassData, perClassMean):
		#note that, we have a perclass data that is for a single class we have a list of input Vector which has [[x1, x2, x3], ...]
		#we will evaluate the covariance matrix with this
		l2=[]
		for i in range(0, len(perClassData[0])):
			l1=[]
			for j in range(0,len(perClassData[0]) ):
				sum=0;
				for k in range(0, len(perClassData[0])):
					sum=sum+(perClassData[k][i]*perClassData[k][j]-perClassMean[i]*perClassMean[j])
				sum=sum/len(perClassData)
				l1.append(sum)
			l2.append(l1)
		return l2

	def covMatrix(mod_training_data, num_features, meanList):
		combinedCovMatrix=[]
		for i in range(0, len(mod_training_data)):
			l1=[];l1.append(mod_training_data[i][0])
			l1.append(mathematica.cov(mod_training_data[i][1], meanList[i][1]))
			combinedCovMatrix.append(l1)
		return combinedCovMatrix