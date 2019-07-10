
# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\diabetes0.csv')
df.head()

X=df.iloc[:,0:-1].values 
y=df.iloc[:,-1].values

# feature extraction
# Extra-trees classifier:
#This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) 
#on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

model = ExtraTreesClassifier()
model.fit(X, y)
print(df.columns)
print(model.feature_importances_)

l=[]
for i in range(len(df.columns)-1):
    l.append((model.feature_importances_[i], df.columns[i]))

print(l,'\n')
l.sort(reverse=True)
print('\n',l)

for i in range(4):
    print(l[i][1])
