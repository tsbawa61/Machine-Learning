
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans

df=pd.read_csv(r'D:\backupKRK2206\PythonProgs\cencusIncomeTmp.csv')

X=df.iloc[:,:8].values
df.columns
len(df)

df2=df.groupby(['earning',' sex'])
print(df2.head())
z1=df2.count()
z2=z1['age']
tFemale=z2[0]+z2[2]
tMale  =z2[1]+z2[3]
pFemale=z2[0]/tFemale
pMale=z2[1]/tMale

gFemale=pFemale*pFemale+(1-pFemale)*(1-pFemale)
gMale=pMale*pMale+(1-pMale)*(1-pMale)

wgGender=(tFemale/len(df)*gFemale)+(tMale/len(df)*gMale)

