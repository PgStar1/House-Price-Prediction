import pandas as pd
from src.preprocessing import preprocess
from src.model import train_model

df = pd.read_csv('data/train.csv')
df_clean = preprocess(df)
rmse = train_model(df_clean)

print(f"RMSE: {rmse}")

with open("results/metrics.txt", "w") as f:
    f.write(f"RMSE: {rmse}")
