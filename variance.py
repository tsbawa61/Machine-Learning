# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import chi2
 
df = pandas.read_csv(r'D:\backupKRK2206\PythonProgs\boston.csv')
print(df.head())
print(df.shape)
print(df.columns)

array = df.values
X = array[:,:13]
Y = array[:,13]

#VarianceThreshold is Feature selector that removes all low-variance features.
#This feature selection algorithm looks only at the features (X), not the desired outputs (y), 
#and can thus be used for unsupervised learning.

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.9 * (1 - .9))) # for 90% threshhold
varTh = sel.fit(X, Y)

numpy.set_printoptions(precision=3)
print(sel.variances_)

for i in range(len(sel.variances_)):
    print(sel.variances_[i], end=" ")
    
print(X.shape)
featTransformed = sel.transform(X)
print(df.columns)
print(featTransformed.shape)

#print(featTransformed[0:5,:])
featBack=sel.inverse_transform(featTransformed)
#print(featBack[0:5,:])

