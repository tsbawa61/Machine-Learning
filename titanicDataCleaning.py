import pandas as pd
import numpy as np
from pandas import ExcelFile
import matplotlib.pyplot as plt


df = pd.read_excel(r'f:\pythonprogs\titanic.xls', sheet_name='titanic3')
print(df.shape)
print(df.columns)

print(df.info())
print(df.describe())
print(df.head())

print(df.isnull().sum())
df1=df.dropna()
print(df1.shape)

# get rid of the useless cols
dropping = ['name', 'ticket','boat','home.dest']
df.drop(dropping,axis=1, inplace=True)
print(df.columns)

# fill missing values with mean column values
df.fillna(df.mean(),inplace=True)
# count the number of NaN values in each column
print(df.isnull().sum())

#pclass : ensure no na contained
print(df.pclass.value_counts(dropna=False))

#Cabin :# checking missing val
# 1014 out of 1309 are missing, drop this col 
# print(df.cabin.isnull().sum())
df.drop('cabin',axis=1,inplace=True)
df.drop('body',axis=1,inplace=True)

#Embark : 2 missing value
print(df.embarked.value_counts())
# fill the majority val,'s', into missing val col
df['embarked'].fillna('S',inplace=True)
print(df.isnull().sum())
print(df.shape)


#Changing categorical data into numerical
from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()

df['sex']=encoder.fit_transform(df['sex'])
df['embarked']=encoder.fit_transform(df['embarked'])
#input("Press Enter")
print(df.corr())

