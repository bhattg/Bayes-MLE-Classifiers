import pandas as pd
import numpy as np
class modifier:
	def modify(df):
		#in this we will separate data of form input as [x1, x2, ..., y] to a mapping of classes. In short separating data on the basis of class
		num_training_set=df.shape[0] 
		num_features= df.shape[1]-1
		num_classes=0
		class_mapping={}
		mod_training_data = []
		#we need to iterate over the dataframe. Add the classes to the dictionary. The key would be classes and the value would be a list of list 
		#the other dictionary will map the numeric values to the actual name of the classes.
		for i in range(0, num_training_set):
		    key=df.iat[i, 0]
		    if key in class_mapping.keys():
		        class_mapping[key].append((df.iloc[i,1:num_features+1]).tolist())
		    else:
		        class_mapping[key]=[(df.iloc[i,1:num_features+1]).tolist()]
		        num_classes=num_classes+1
		for keys in class_mapping.keys():
		    l1= [keys, (class_mapping[keys])]; mod_training_data.append(l1)
		return mod_training_data
