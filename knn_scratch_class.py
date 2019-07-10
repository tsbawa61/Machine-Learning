def train(X_train, y_train):
	# do nothing 
	return
def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)
	print(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])
	
	# return most common target
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
		
	# train on the input data
	train(X_train, y_train)

	# predict for each testing observation
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))

import pandas as pd
# loading training data
df = pd.read_csv(r'f:\PythonProgs\iris.csv')
df.head()
        # define column names
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'spieces']

# loading libraries
import numpy as np
from sklearn.model_selection import train_test_split

# create input matrix X and target vector y
X=df.iloc[:,0:-1].values 
y=df.iloc[:,-1].values

# making our predictions 
predictions = []
from collections import Counter

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

kNearestNeighbor(X_train, y_train, X_test, predictions, 7)
predictions = np.asarray(predictions)
print(predictions)

# evaluating accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions) * 100
print('\nThe accuracy of OUR classifier is %d%%' % accuracy)

