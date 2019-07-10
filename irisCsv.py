#pip install xlrd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
 
from pandas import DataFrame, read_csv
import pandas as pd
file = r'f:\pythonprogs\iris.csv'
df = pd.read_csv(file)
print(df)

print(type(df))
print(df['sepal width'][0])
#print(df.head())
#print(df.tail())
#print(df.columns)
#print(df.shape)
#print(df.shape[0])
print(df['spieces'].unique())
print(df.groupby('spieces').size()) #species distribution
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
X = df[feature_names]
print(X)
y = df['spieces']
print(y)



sepalWidth = df['sepal width']
#print("sepalWidth\n",sepalWidth)

sepalLength = df['sepal length']
#print("sepalLength\n",sepalLength)
#petalLength = df[petal length']
train_input=[]
train_output=[]
for i in range(df.shape[0]):
        a=df['sepal length'][i]
        b=df['sepal width'][i]
        c=df['petal length'][i]
        d=df['petal width'][i]
        train_input.append([a,b,c,d])
        e=df['spieces'][i]
        train_output.append(e)
#print(train_input)
#print(train_output)

P=[[4.9,3,1.4,.2]]
P=[[6.4,3.2,4.5,1.5]]
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
