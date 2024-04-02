#this is a classifications technique
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(xtest,ytest)
tree.plot_tree(clf,feature_names=iris.feature_names,class_names=iris.target_names,filled = True)
plt.show()