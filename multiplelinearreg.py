import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data={
    'area':[1200,1500,1800,2000,1000],
    'bedroom':[1,2,3,4,2],
    'age':[10,5,8,3,15],
    'prices':[25000,30000,32000,4000,20000]
}

df = pd.DataFrame(data)
x = df[['area','bedroom','age']]
y = df.prices

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)
print(ypred)
print("\n")
mse = mean_squared_error(ytest,ypred)
print(mse)

