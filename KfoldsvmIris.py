import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\iris.csv')
X=df.iloc[:,0:-1].values 
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn import svm
# we create an instance of SVM and fit out data.
# We do not scale our data since we want to plot the support vectors
svc = svm.SVC()
svc.fit(X_train, y_train)  
y_pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy of Test Data is',accuracy_score(y_pred, y_test))

from sklearn.model_selection import cross_val_score
#The code below perform K-Fold Cross Validation on our random forest model, using 10 folds (K = 10). 
#Therefore it outputs an array with 10 different scores.

scores = cross_val_score(svc, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
