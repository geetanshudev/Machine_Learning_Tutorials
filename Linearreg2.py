import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

iris = pd.read_csv("C:\\Users\\Geetanshu Dev\\Documents\\csv\\iris.csv")
print(iris.columns)

x = iris[['sepallength']]
y = iris[['petallength']]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

lr = LinearRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)
'''
plt.scatter(x,y,color = 'blue')
plt.plot(xtest,ypred,color = 'red')
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
'''
#now we have see that with mean_squared_error

print("Mean Square error = ",mean_squared_error(ytest,ypred))
print("R2 score = ",r2_score(ytest,ypred))
