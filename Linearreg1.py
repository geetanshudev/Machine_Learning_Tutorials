import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(100,1)*10
y = 2*x*3+np.random.rand(100,1)*2

'''
plt.scatter(x,y,color = 'blue')
plt.xlabel("House Size")
plt.ylabel("Price")
plt.show()
print(x)
print("\n")
print(y)
'''
model = LinearRegression()
model.fit(x,y)
xtest = np.array([[2],[4],[5]])
pred = model.predict(xtest)

plt.scatter(x,y,color='Blue')
plt.plot(xtest,pred,linewidth = 5,color = 'red',linestyle = 'solid')
plt.xlabel("House price")
plt.ylabel("Price")
plt.show()