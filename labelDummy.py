import pandas as pd
# making data frame from csv file 
df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\nba.csv') 

df['Team'].unique()
df['Team'].unique().shape

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Team1'] = labelencoder.fit_transform(df['Team'])


df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\iris.csv') 
dummy=pd.get_dummies(df['spieces'])
dummy.head()

df=pd.concat([df, dummy], axis=1)

