import pandas as pd
import os

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
os.makedirs("data/raw", exist_ok=True)

df = pd.read_csv(url)
df.to_csv("data/raw/titanic.csv", index=False)

print("Dataset downloaded")
