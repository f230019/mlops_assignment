import pandas as pd
import joblib
import os

df = pd.read_csv("features/features.csv")
X = df.drop("Survived", axis=1)

model = joblib.load("models/svm_model.pkl")
predictions = model.predict(X)

results = pd.DataFrame({"Prediction": predictions})

os.makedirs("results", exist_ok=True)
results.to_csv("results/predictions.csv", index=False)

print(" Predictions generated")
