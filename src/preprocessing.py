import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.select_dtypes(include=['int64', 'float64']).dropna(axis=1)
    df = df.drop(['Id'], axis=1)
    return df
