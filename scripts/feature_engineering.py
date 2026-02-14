import pandas as pd
import os

df = pd.read_csv("data/processed/cleaned.csv")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

features = df[
    ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Survived"]
]

os.makedirs("features", exist_ok=True)
features.to_csv("features/features.csv", index=False)

print("Features engineered")
