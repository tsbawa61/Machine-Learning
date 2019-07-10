import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\iris.csv')
X=df.iloc[:,0:-1].values 
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy of Test Data is',accuracy_score(y_pred, y_test))

# Summary of the predictions made by the classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("#===== confusion matrix")
print(confusion_matrix(y_test, y_pred ))

#Plot Confusion Matrix
from sklearn import metrics
#pip install scikit-plot
import scikitplot
from matplotlib import pyplot as plt

scikitplot.metrics.plot_confusion_matrix(y_test, y_pred)
#scikitplot.metrics.plot_confusion_matrix(y_test, y_pred,normalize=True)
plt.show()

print("\nclassification_report:",classification_report(y_test, y_pred))


from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(classifier, out_file=r'd:\irisForest.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

