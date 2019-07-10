def entrpy(z,tot):
    l=[]
    for i in range(len(z)) :
        l.append((i,z.iloc[i]/tot)) 
        
    E=0
    for i in range(len(z)):
        p=l[i][1]
        print(z[i],':',p)
        E=E+(-1)*p*math.log2(p)
    return E

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans

df=pd.read_csv(r'D:\backupKRK2206\PythonProgs\cencusIncomeTmp.csv')

X=df.iloc[:,:8].values
df.columns

df1=df.groupby('earning')
#print(df1.head())
y=df1.count()
#z=y['age']
z=y.iloc[:,0]
eEarning=entrpy(z,len(df))

df2=df.groupby(['earning',' sex'])
#print(df2.head())
z1=df2.count()
z2=z1.iloc[:,0]


l1=[]
l2=[]
k=0
for x in z2:
    k=k+1
    if (k%2==0):
        l1.append(x)
    else:
        l2.append(x)
        
l1=pd.Series(l1)
l2=pd.Series(l2)
eEarningMale   = entrpy(l1,l1.sum())
eEarningFemale = entrpy(l2,l2.sum())

eEarningSex=eEarningMale+eEarningFemale

gEarningSex=eEarning-eEarningSex

