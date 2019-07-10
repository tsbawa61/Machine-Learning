from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
#df = pd.read_excel(r'D:\backupKRK2206\PythonProgs\iris.xlsx', sheet_name='iris')
df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\diabetes_clean.csv')
print(df.shape)

X = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

print(len(Xtrain),len(Xtest))


import scikitplot as skplt

#{using RandomForestClassifier}
clf = RandomForestClassifier()
clf = clf.fit(X=Xtrain,y=ytrain)


#from sklearn.model_selection import learning_curve

skplt.estimators.plot_learning_curve(clf,Xtrain,ytrain,title='Learning Curve for Random Forest ()')

#{Decision Tree Model}
clf = DecisionTreeClassifier()
clf = clf.fit(X=Xtrain,y=ytrain)

skplt.estimators.plot_learning_curve(clf,Xtrain,ytrain,title='Learning Curve for DecisionTreeClassifier ()')
clf = DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X=Xtrain,y=ytrain)

skplt.estimators.plot_learning_curve(clf,Xtrain,ytrain,title='Learning Curve for DecisionTreeClassifier (max_depth=4)')

#{K Neighbors Classifier}
clf = KNeighborsClassifier()
clf=clf.fit(X=Xtrain,y=ytrain)

skplt.estimators.plot_learning_curve(clf,Xtrain,ytrain,title='Learning Curve for KNeighborsClassifier ()')
plt.show()
