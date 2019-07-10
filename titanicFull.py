import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt

# machine learning Classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


df = pd.read_excel(r'f:\pythonprogs\titanic.xls', sheet_name='titanic3')
print(df.shape)
print(df.columns)

"""print(df.info())
print(df.describe())
print(df.head())"""

# show the overall survival rate (38.38), as the standard when choosing the fts
print('Overall Survival Rate:',df['survived'].mean())
print(df.isnull().sum())
df1=df.dropna()
print(df1.shape)

# get rid of the useless cols
dropping = ['name', 'ticket','boat','home.dest']
df.drop(dropping,axis=1, inplace=True)
print(df.columns)

# fill missing values with mean column values
df.fillna(df.mean(),inplace=True)
# count the number of NaN values in each column
print(df.isnull().sum())

#pclass : ensure no na contained
print(df.pclass.value_counts(dropna=False))

#Cabin :# checking missing val
# 1014 out of 1309 are missing, drop this col 
# print(df.cabin.isnull().sum())
df.drop('cabin',axis=1,inplace=True)
df.drop('body',axis=1,inplace=True)

#Embark : 2 missing value
print(df.embarked.value_counts())
# fill the majority val,'s', into missing val col
df['embarked'].fillna('S',inplace=True)
print(df.isnull().sum())
print(df.shape)


#Changing categorical data into numerical
from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()

df['sex']=encoder.fit_transform(df['sex'])
df['embarked']=encoder.fit_transform(df['embarked'])
#input("Press Enter")
print(df.corr())

# Visualizations

print(df.groupby('pclass').mean())
class_sex_grouping = df.groupby(['pclass','sex']).mean()
print(class_sex_grouping)

class_sex_grouping['survived'].plot.bar()

print(df.corr())
# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('scatter-matrix')
plt.show()

group_by_age = pd.cut(df["age"], np.arange(0, 90, 10))
age_grouping = df.groupby(group_by_age).mean()
print(age_grouping)

age_grouping['survived'].plot.bar()
plt.show()
# box and whisker plots
feature_names = ['pclass', 'sex',  'embarked']
df[feature_names].plot(kind='box', sharex=False, sharey=False,title='Box Plot',subplots=True)

# boxplot on each feature split out by Sex
feature_names = ['pclass', 'age', 'fare', 'embarked']

df.boxplot(column=feature_names,by="sex",figsize=(10,10))
plt.suptitle('Box Plot by Sex Code (0-Female, 1-Male)')


df.plot.scatter(x='fare', y='age');

# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('scatter-matrix')
plt.show()

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


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)


print("KNeighborsClassifier  Accuracy : ", acc_knn)

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

print("Decision Tree Accuracy : ", acc_decision_tree)


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print("Random Forest  Accuracy : ", acc_random_forest)

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)

print("Logistic Regression  Accuracy : ", acc_log)

###########################################################################
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
