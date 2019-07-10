import pandas as pd
# making data frame from csv file 
df = pd.read_csv(r'D:\backupKRK2206\PythonProgs\cencusIncome.csv') 
df.columns
print(df[' race'].unique())
df[' race'].unique().shape

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df[' race1'] = labelencoder.fit_transform(df[' race'])


from sklearn.preprocessing import LabelBinarizer
lb = preprocessing.LabelBinarizer()
lb.fit(df[' race1'])
lb.classes_
lb.y_type_

a=lb.transform(df[' race1'])
dfBinRace=pd.DataFrame(a,columns=[   ' Amer-Indian-Eskimo',' Asian-Pac-Islander',' Black', ' Other',' White'])

print(df[' sex'].unique())
df[' sex'].unique().shape

lb = preprocessing.LabelBinarizer()
lb.fit(df[' sex'])
lb.classes_

b=lb.transform(df[' sex'])
dfBinSex=pd.DataFrame(b ,columns=[ ' Gender'])
