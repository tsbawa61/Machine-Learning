import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt

# machine learning Classifiers

from sklearn.svm import SVC, LinearSVC

df = pd.read_excel(r'f:\pythonprogs\titanic.xls', sheet_name='titanic3')
print(df.shape)
print(df.columns)

X = df.drop("survived", axis=1)
y = df["survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Support Vector Machines

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print("Support Vector Machines Accuracy : ", acc_svc)


# Summary of the predictions made by the classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier=knn
classifier.fit(X=X_train,y=y_train)

y_pred = classifier.predict(X_test)

# Accuracy score
from sklearn.metrics import accuracy_score
print('Accuracy of Test Data is',accuracy_score(y_pred, y_test))

print("#===== Summary of the predictions made by the classifier")

print("No of Test Cases",len(y_test))

print("#===== confusion matrix")
label_lst=[0,  1]
print("Label List for Confusion Matrix: ")
print(label_lst)
print(confusion_matrix(y_test, y_pred,labels=label_lst))

print("\nclassification_report:",classification_report(y_test, y_pred))

#Plot Confusion Matrix
from sklearn import metrics
#pip install scikit-plot
import scikitplot
from matplotlib import pyplot as plt

scikitplot.metrics.plot_confusion_matrix(y_test, y_pred)
#scikitplot.metrics.plot_confusion_matrix(y_test, y_pred,normalize=True)
plt.show()

#scatter plot matrix for predictions vs actual values
X_test1=range(len(y_test))
plt.scatter(X_test1,y_test,color='red', marker='<')
plt.scatter(X_test1,y_pred,color='yellow', marker='>')
#plt.plot(X_test1,y_pred, 'ro')
#plt.plot(X_test1,y_pred, color='red',linewidth=2)
plt.show()
