import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')

# Overview
print(df.shape)
print(df.info())

# Correlation with SalePrice
corr = df.corr()['SalePrice'].sort_values(ascending=False)
print(corr.head(10))

# Visuals
sns.histplot(df['SalePrice'], kde=True)
sns.boxplot(x='OverallQual', y='SalePrice', data=df)