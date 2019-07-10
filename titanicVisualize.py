import pandas as pd
import numpy as np
from pandas import ExcelFile
import matplotlib.pyplot as plt

df = pd.read_excel(r'f:\pythonprogs\titanic.xls', sheet_name='titanic3')
# Visualizations

print(df.groupby('pclass').mean())
class_sex_grouping = df.groupby(['pclass','sex']).mean()
print(class_sex_grouping)

class_sex_grouping['survived'].plot.bar()

print(df.corr())

# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('scatter-matrix')
plt.show()

group_by_age = pd.cut(df["age"], np.arange(0, 90, 10))
age_grouping = df.groupby(group_by_age).mean()
print(age_grouping)

age_grouping['survived'].plot.bar()
plt.show()

# box and whisker plots
feature_names = ['pclass', 'sex',  'embarked']
df[feature_names].plot(kind='box', sharex=False, sharey=False,title='Box Plot',subplots=True)

# boxplot on each feature split out by Sex
feature_names = ['pclass', 'age', 'fare', 'embarked']

df.boxplot(column=feature_names,by="sex",figsize=(10,10))
plt.suptitle('Box Plot by Sex Code (0-Female, 1-Male)')


df.plot.scatter(x='fare', y='age');

# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10))
plt.suptitle('scatter-matrix')
plt.show()

X = df.drop("survived", axis=1)
y = df["survived"]

