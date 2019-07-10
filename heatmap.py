import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\iris.csv')

dfcorr=df.corr()

print(dfcorr)

#Heatmap plots rectangular data as a color-encoded matrix.
#Input is 2D dataset that can be coerced into an ndarray. 
#If a Pandas DataFrame is provided, the index/column information will be used to label the columns and rows.

sns.heatmap(dfcorr)

plt.figure(figsize=(8,8))
sns.heatmap(dfcorr, annot=True, linewidth=0.5)

sns.heatmap(dfcorr, annot=True, linewidth=0.5, cmap='coolwarm')
sns.heatmap(dfcorr, annot=True, linewidth=0.5, cmap='YlGnBu')
sns.heatmap(dfcorr, annot=True, linewidth=0.5, yticklabels=False)
sns.heatmap(dfcorr, annot=True, linewidth=0.5, xticklabels=False)
sns.heatmap(dfcorr, annot=True, linewidth=0.5, cbar=False)

from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('scatter-matrix')
plt.show()

# boxplot on each feature split out by Sex
feature_names = ['sepal length', 'sepal width']

df.boxplot(column=feature_names,by="spieces",figsize=(10,10))
plt.suptitle('Box Plot by spieces ')

X=df.iloc[:,0:-1].values 
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)


print("KNeighborsClassifier  Accuracy : ", acc_knn)

#scatter plot matrix for predictions vs actual values
X_test1=range(len(y_test))
plt.scatter(X_test1,y_test,color='red', marker='<')
plt.scatter(X_test1,y_pred,color='yellow', marker='>')
plt.show()

# Summary of the predictions made by the classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Accuracy score
from sklearn.metrics import accuracy_score
print('Accuracy of Test Data is',accuracy_score(y_pred, y_test))

print("#===== Summary of the predictions made by the classifier")

print("No of Test Cases",len(y_test))

print("#===== confusion matrix")
print(confusion_matrix(y_test, y_pred ))

#print(df['spieces'].unique())

label_lst=['Iris-setosa' ,'Iris-virginica','Iris-versicolor']
print("Label List for Confusion Matrix: ", label_lst,'\n')
print(confusion_matrix(y_test, y_pred,labels=label_lst))

label_lst=['Iris-setosa' ,'Iris-virginica']
print("Label List for Confusion Matrix: ", label_lst,'\n')
print(confusion_matrix(y_test, y_pred,labels=label_lst))


#Plot Confusion Matrix
from sklearn import metrics
#pip install scikit-plot
import scikitplot
from matplotlib import pyplot as plt

scikitplot.metrics.plot_confusion_matrix(y_test, y_pred)
#scikitplot.metrics.plot_confusion_matrix(y_test, y_pred,normalize=True)
plt.show()

print("\nclassification_report:",classification_report(y_test, y_pred))
#Plotting Classification Report

classRep=classification_report(y_test, y_pred)
l=classRep.splitlines()
headings=l[0].split()
headings=['prediction']+headings
headings2=l[1].split()
l1=[]
for i in range(2,len(l) -2):
    l3=l[i].split()
    l4=[l3[0]]
    l4=l4+[float(x) for x in l3[1:]]
    l1.append(l4)

l2=[]
l2.append(headings)
for i in range(2,len(l) -2):
        l2.append(list(l1[i-2])) 

d={}
d[l2[0][1]]=[x[1] for x in l2[1:]]
d[l2[0][2]]=[x[2] for x in l2[1:]]
d[l2[0][3]]=[x[3] for x in l2[1:]]
d[l2[0][4]]=[x[4] for x in l2[1:]]

lbl=[x[0] for x in l2[1:]] 
dfClassRep=pd.DataFrame(d)
plt.figure(figsize=(8,8))
sns.heatmap(dfClassRep, annot=True, linewidth=0.5,yticklabels=lbl)



