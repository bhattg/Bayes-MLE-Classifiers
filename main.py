from modifyInput import modifier
from maths import mathematica
import pandas as pd 
from naiveBayes import naiveBayes
from pathlib import Path 
from bayes import bayes
import numpy as np

link = "Medical_data.csv"
testframe= pd.read_csv("T.csv")
totalTestpoints=testframe.shape[0]

classifier = naiveBayes(link)	
classifier.fit()

count=0
for i in range(0, totalTestpoints):
	
	original=testframe.iat[i, 0]
	predicted=classifier.predict((testframe.iloc[i, 1:4]).tolist())
	if predicted==original:
		count=count+1

print("accuracy=", 100*count/totalTestpoints)
