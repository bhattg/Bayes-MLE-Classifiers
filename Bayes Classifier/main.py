import pandas as pd  
from Bayes import Bayes

t= pd.read_csv("test_medical.csv")
classifier = Bayes("Medical_data.csv")	
classifier.fit()
count=0
totalTestpoints=t.shape[0]
for i in range(0, totalTestpoints):	
	original=t.iat[i, 0]
	predicted=classifier.predict((t.iloc[i, 1:4]).tolist())
	if predicted==original:
		count=count+1
	
print("accuracy=", 100*count/totalTestpoints)