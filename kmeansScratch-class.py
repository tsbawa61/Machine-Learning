import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, cross_validation
import pandas as pd

##X = np.array([[1, 2],
##              [1.5, 1.8],
##              [5, 8],
##              [8, 8],
##              [1, 0.6],
##              [9, 11]])
##
##
##colors = ['r','g','b','c','k','o','y']



#K_Means(k=2, tol=0.001, max_iter=300)
        
def fit(data,k=2, tol=0.001, max_iter=300):

        global centroids #= {}

        for i in range(k):
            centroids[i] = data[i]

        for i in range(max_iter):
            classifications = {}

            for i in range(k):
                classifications[i] = []

            for featureset in X:
                distances = [np.linalg.norm(featureset-centroids[centroid]) for centroid in centroids]
                classification = distances.index(min(distances))
                classifications[classification].append(featureset)

            prev_centroids = dict(centroids)

            for classification in classifications:
                centroids[classification] = np.average(classifications[classification],axis=0)

            optimized = True

            for c in centroids:
                original_centroid = prev_centroids[c]
                current_centroid = centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

def predict(data):
        distances = [np.linalg.norm(data-centroids[centroid]) for centroid in centroids]
        classification = distances.index(min(distances))
        return classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r'D:\backupKRK2206\PythonProgs\iris.csv')

X=df.iloc[:,[0,1,2,3]].values

fit(X)
"""
correct = 0
for i in range(len(X)):

    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1


print(correct/len(X))
"""