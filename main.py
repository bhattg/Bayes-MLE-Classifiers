from modifyInput import modifier
from naiveBayes import naiveBayes
import pandas as pd

link = "Medical_data.csv"
testframe= pd.read_csv("test_medical.csv")
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