from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
 
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\iris.csv')

X = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)
np.linspace(.1, 1.0, 5)
print(len(X))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

import scikitplot as skplt

#{K Neighbors Classifier}
knn = KNeighborsClassifier()
knn=knn.fit(X=Xtrain,y=ytrain)

clf=knn
yprobas_knn = clf.predict_proba(Xtest)
skplt.metrics.plot_roc(ytest, yprobas_knn, title='ROC Curve for KNeighborsClassifier()')

#Random Forest
rfor = RandomForestClassifier()
rfor=rfor.fit(X=Xtrain,y=ytrain)

clf=rfor

plt.figure(figsize=(10,10))
print(len(Xtest))
yprobas_rf = clf.predict_proba(Xtest)
skplt.metrics.plot_roc(ytest, yprobas_rf, title='ROC Curve for RandomForestClassifier()')


#{Decision Tree Model}
clf = DecisionTreeClassifier()
clf = clf.fit(X=Xtrain,y=ytrain)

yprobas_dt = clf.predict_proba(Xtest)
skplt.metrics.plot_roc(ytest, yprobas_dt, title='ROC Curve for DecisionTreeClassifier()')


plt.show()

#SVC
from sklearn import svm

clf = svm.SVC(probability=True)
clf = clf.fit(X=Xtrain,y=ytrain)

yprobas_dt = clf.predict_proba(Xtest)

from sklearn.metrics import roc_curve, auc
yprobas_dt = clf.predict_proba(Xtest)
y_pred = clf.predict(Xtest)


#Compute Receiver operating characteristic (ROC)
fpr, tpr, thresholds = roc_curve(y_pred, ytest)
"""
#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
"""
