#pip install xlrd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
 
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
df = pd.read_excel(r'f:\pythonprogs\iris.xlsx', sheet_name='iris')
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
X = df[feature_names]
#print(X)
y = df['spieces']
#print(y)

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


#Performance of the Classifier
X=train_input;y=train_output

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Summary of the predictions made by the classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


classifier=knn
classifier.fit(X=X_train,y=y_train)
print ("Using Chosen Classifier the Prediction is " + str(classifier.predict(P)) +"\n")

y_pred = classifier.predict(X_test)

# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))

print("#===== Summary of the predictions made by the classifier")

print("No of Test Cases",len(y_test))

print()
for i in range(len(y_test)):
        if (y_test[i]!=y_pred[i]):
                print("\n Test Case :",y_test[i],"Prediction: ",y_pred[i])

print("#===== confusion matrix")
label_lst=["Iris-setosa",  "Iris-versicolor", "Iris-virginica"]
print("Label List for Confusion Matrix: ")
print(label_lst)
print(confusion_matrix(y_test, y_pred,labels=label_lst))

print("\nclassification_report:",classification_report(y_test, y_pred))


input("Press Enter")

#Visualizations
from matplotlib import pyplot as plt

# more info on the data
print(df.info())
print(df['sepal length'].unique()); print(df['sepal width'].unique())
print(df['petal length'].unique());print(df['petal width'].unique())

# histograms
df.hist(edgecolor='black', linewidth=1.2)
plt.suptitle('Histogram')

# box and whisker plots
df.plot(kind='box', sharex=False, sharey=False,title='Box Plot')

# boxplot on each feature split out by species
df.boxplot(by="spieces",figsize=(10,10))
plt.suptitle('Box Plot by Spieces')

# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('scatter-matrix')
plt.show()

#scatter plot matrix for predictions vs actual values
X_test1=range(len(y_test))
plt.scatter(X_test1,y_test,color='blue', marker='^')
plt.plot(X_test1,y_pred, 'ro')
#plt.plot(X_test1,y_pred, color='red',linewidth=2)
plt.show()

#Plot Confusion Matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#pip install scikit-plot
import scikitplot
from matplotlib import pyplot as plt

print(classification_report(y_test, y_pred))
scikitplot.metrics.plot_confusion_matrix(y_test, y_pred)
#scikitplot.metrics.plot_confusion_matrix(y_test, y_pred,normalize=True)
plt.show()

#print(type(df))
#print(df['sepal width'][0])
#print(df.head())
#print(df.tail())
#print(df.columns)
#print(df.shape)
#print(df.shape[0])
#print(df['spieces'].unique())
#print(df.describe())
#print(df.groupby('spieces').size()) #species distribution
#sepalWidth = df['sepal width']
#print("sepalWidth\n",sepalWidth)

#sepalLength = df['sepal length']
#print("sepalLength\n",sepalLength)
#petalLength = df[petal length']
#print(train_input)
#print(train_output)
