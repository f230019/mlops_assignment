import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import os

df = pd.read_csv("features/features.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_model.pkl")

print("Model trained and saved")
