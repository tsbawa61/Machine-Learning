# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
 
df = pandas.read_csv(r'D:\backupKRK2206\PythonProgs\diabetes0.csv')
print(df.head())
array = df.values
X = array[:,0:8]
Y = array[:,8]

# feature extraction
#Select features according to the k highest scores
#The chi-square test measures dependence between , so using this function “weeds out” 
#the features that are the most likely to be independent of class and therefore irrelevant for classification.

bestK = SelectKBest(score_func=chi2, k=4)
fitKbest = bestK.fit(X, Y)

# summarize scores
numpy.set_printoptions(precision=3)
print(df.shape)
print(df.columns)
print(fitKbest.scores_)

l=[]
for i in range(len(df.columns)-1):
    l.append((fitKbest.scores_[i], df.columns[i]))

print(l,'\n')
l.sort(reverse=True)
print('\n',l)

for i in range(4):
    print(l[i][1])

print(fitKbest.pvalues_)
featTransformed = fitKbest.transform(X)

# summarize selected features
print(featTransformed[0:5,:])
featBack=fitKbest.inverse_transform(featTransformed)
print(featBack[0:5,:])
