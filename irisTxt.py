#pip install xlrd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
 
from pandas import DataFrame, read_csv
import pandas as pd
df = pd.read_table(r'f:\pythonProgs\iris.txt')

print(df)

print(type(df))
print(df['sepal_width'][0])
#print(df.head())
#print(df.tail())
#print(df.columns)
#print(df.shape)
#print(df.shape[0])
print(df['spieces'].unique())
print(df.groupby('spieces').size()) #species distribution
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[feature_names]
print(X)
y = df['spieces']
print(y)



sepalWidth = df['sepal_width']
#print("sepalWidth\n",sepalWidth)

sepalLength = df['sepal_length']
#print("sepalLength\n",sepalLength)
#petalLength = df[petal length']
train_input=[]
train_output=[]
for i in range(df.shape[0]):
        a=df['sepal_length'][i]
        b=df['sepal_width'][i]
        c=df['petal_length'][i]
        d=df['petal_width'][i]
        train_input.append([a,b,c,d])
        e=df['spieces'][i]
        train_output.append(e)
#print(train_input)
#print(train_output)

P=[[4.9,3,1.4,.2]]
#P=[[6.4,3.2,4.5,1.5]]
#P=[[7.6,3,6.6,2.1]]

#{Decision Tree Model}
clf = DecisionTreeClassifier()
clf = clf.fit(X=train_input,y=train_output)
print ("\n1) Using Decision Tree Prediction is " + str(clf.predict(P)))
 
#{K Neighbors Classifier}
knn = KNeighborsClassifier()
knn.fit(X=train_input,y=train_output)
print ("2) Using K Neighbors Classifier Prediction is " + str(knn.predict(P)))
 
 
#{using RandomForestClassifier}
rfor = RandomForestClassifier()
rfor.fit(X=train_input,y=train_output)
print ("3) Using RandomForestClassifier Prediction is " + str(rfor.predict(P)) +"\n")
