#in this we create a dicision tree of iris data and measure its accuracy 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
data = load_iris()
x = data.data
y = data.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)

model = DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
acc = accuracy_score(ypred,ytest)
print(acc)
conf = confusion_matrix(ypred,ytest)
print(conf)