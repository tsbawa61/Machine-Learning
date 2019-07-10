import pandas as pd
# loading training data
df = pd.read_csv(r'f:\PythonProgs\iris.csv')
df.head()

# loading libraries
import numpy as np
from sklearn.model_selection import train_test_split

# create design matrix X and target vector y
X=df.iloc[:,0:-1].values 
y=df.iloc[:,-1].values

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# loading library
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
from sklearn.metrics import accuracy_score

print (accuracy_score(y_test, pred))

