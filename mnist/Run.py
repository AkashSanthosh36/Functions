#importing required packages

import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

#Reading the data
bc=pd.read_csv('train.csv').values

clf=DecisionTreeClassifier()

#Training Datasets
xtrain, ytrain=bc[:21000,1:], bc[:21000,0:1]

#Testing Datasets
xtest, ytest=bc[21000:,1:], bc[21000:,0:1]

clf.fit(xtrain,ytrain)
d=xtest[8]
d.shape=(28,28)
plt.imshow(255-d,cmap='gray')
print(clf.predict([xtest[8]]))
plt.show()

#Accuracy
count=0
p=clf.predict(xtest)
for i in range(21000):
	if(p[i]==ytest[i]):
		count+=1
print("Accuracy:",(count/21000)*100)		

"""#Reading the image
img=Image.open("7.jpg")
img=img.resize((28,28))
#img.show()"""