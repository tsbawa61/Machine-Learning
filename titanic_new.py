"""Titanic Data Set column heading variables have the following meanings:

survival: Survival (0 = no; 1 = yes)
class: Passenger class (1 = first; 2 = second; 3 = third)
name: Name
sex: Sex
age: Age
sibsp: Number of siblings/spouses aboard
parch: Number of parents/children aboard
ticket: Ticket number
fare: Passenger fare
cabin: Cabin
embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat: Lifeboat (if survived)
body: Body number (if did not survive and body was recovered)
"""

#pip install xlrd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
 
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt

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
print(df.groupby('pclass').mean())
class_sex_grouping = df.groupby(['pclass','sex']).mean()
print(class_sex_grouping)

class_sex_grouping['survived'].plot.bar()

group_by_age = pd.cut(df["age"], np.arange(0, 90, 10))
age_grouping = df.groupby(group_by_age).mean()
print(age_grouping)

age_grouping['survived'].plot.bar()
plt.show()

X = df.drop("Survived", axis=1)
y = df["Survived"]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Logistic Regression


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


