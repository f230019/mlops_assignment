import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

df = pd.read_csv("data/raw/titanic.csv")

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned.csv", index=False)

print("Data preprocessed")
