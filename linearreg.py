import joblib
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris.data
y = iris.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

model = GaussianNB()
model.fit(xtrain,ytrain)
'''
ypred = model.predict(xtest)
print(ypred)

acc = accuracy_score(ytest,ypred)
print(acc)
'''

#now we have to save our model to a separate file i.e joblib
file = "NB_modeliris.joblib"
joblib.dump(model,file)

#now we load joblib 
filename = "NB_modeliris.joblib"
NBmodel = joblib.load(filename)

ypred = NBmodel.predict(xtest)
print(ypred)
acc = accuracy_score(ytest,ypred)
print(acc)