import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = pd.read_csv("features/features.csv")["Survived"]
y_pred = pd.read_csv("results/predictions.csv")["Prediction"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1-score: {f1}\n")

print(" Evaluation metrics saved")
