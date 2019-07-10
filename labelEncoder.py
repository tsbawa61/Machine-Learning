import pandas as pd
# making data frame from csv file 
df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\cencusIncome.csv') 
df.columns
print(df[' race'].unique())
df[' race'].unique().shape

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df[' race1'] = labelencoder.fit_transform(df[' race'])


#Changing continous data into Categorical

import pandas as pd
# making data frame from csv file 
df = pd.read_csv(r'f:\PythonProgs\diabetes.csv') 

ranges=[0,18,30,50,60,80,120]
range_names=['Children','Youth','Middle Age','Senior','Retired','Eighty+']

df['age_cat'] =pd.cut(df['Age'], bins=ranges, labels=range_names)

#Dummies Concept
dummy=pd.get_dummies(df['Sex'])
print(type(dummy))
print(dummy.head())

df=pd.concat([df,dummy], axis=1) #Joining df and dummy
df.head()
df.drop('Sex', axis=1,inplace=True)

#conda install -c anaconda pandas-profiling
#pip install pandas-profiling
#import pandas_profiling
#df.profile_report(style={'full_width':True})