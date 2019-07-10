import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import linear_model

df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\diabetes0.csv')
df.head()

X=df.iloc[:,0:-1].values 
y=df.iloc[:,-1].values

regr = linear_model.LinearRegression()

estimator = linear_model.LinearRegression()
#Feature ranking with recursive feature elimination.

#Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), 
#the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller 
#and smaller sets of features. First, the estimator is trained on the initial set of features 
#and the importance of each feature is obtained either through a coef_ attribute 
#or through a feature_importances_ attribute. 
#Then, the least important features are pruned from current set of features.
#That procedure is recursively repeated on the pruned set until the desired number of features to 
#select is eventually reached.

selector = RFE(estimator, 4) # 4 is the number of features to select

selector.fit(X, y)
print(selector.n_features_)
print(selector.support_ )
print(selector.ranking_)

p=selector.transform(X)
q=selector.inverse_transform(p)

l=[]
for i in range(len(df.columns)-1):
    l.append((selector.ranking_[i], df.columns[i]))

print(l,'\n')
l.sort()
print('\n',l)

for i in range(4):
    print(l[i][1])

