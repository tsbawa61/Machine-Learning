#pip install xlrd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
 
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
df = pd.read_excel(r'f:\pythonprogs\iris.xlsx', sheet_name='iris')
print(df.corr())
