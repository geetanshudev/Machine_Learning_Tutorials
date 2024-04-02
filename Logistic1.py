import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x,y = make_classification(n_samples=100,n_features=2,n_informative=2,n_redundant=0,random_state=42)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
model = LogisticRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
print(ypred)
print(ytest)

from sklearn.metrics import accuracy_score,classification_report
acc = accuracy_score(ypred,ytest)
print(acc)
acc1 = classification_report(ypred,ytest)
print(acc1)