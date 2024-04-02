import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
data = load_breast_cancer()
x = data.data
y = data.target
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

model = GaussianNB()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
acc = accuracy_score(ypred,ytest)
print(acc*100)
conf = confusion_matrix(ypred,ytest)
print(conf)


