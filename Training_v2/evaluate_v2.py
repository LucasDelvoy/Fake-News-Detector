import pandas as pd
import numpy as np
import joblib
import torch
from model_v2 import Model
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix

# Functions
def remove_stopwords(text):
    return " ".join([w for w in str(text).split() if w not in stopwords_list])

# Get assets
df = pd.read_csv("./datasets/cleaned_data.csv")
model = Model()
data = joblib.load("./output/data.pkl")
vec_text = joblib.load("./output/vec_text.pkl")
vec_title = joblib.load("./output/vec_title.pkl")
stopwords_list = set(stopwords.words("english"))

# Get data into arrays
X_test_array = data["X_test"].toarray()
X_test = torch.FloatTensor(X_test_array)
y_test = torch.FloatTensor(data["y_test"].values.copy()).view(-1, 1)

# Evaluation
model.load_state_dict(torch.load("output/model.pth"))
model.eval()

with torch.no_grad():
    raw_predictions = model(X_test)
    y_pred = (raw_predictions > 0.5).int()

y_test_np = y_test.flatten()
y_pred_np = y_pred.flatten()

print(classification_report(y_test, y_pred_np))
print(confusion_matrix(y_test, y_pred_np))

test_index = data["y_test"].index
mask_fn = (y_test_np == 1) & (y_pred_np == 0)
false_negative = df.iloc[test_index[mask_fn]]

mask_fp = (y_test_np == 0) & (y_pred_np == 1)
false_positive = df.iloc[test_index[mask_fp]]

print("\n --- EXAMPLES OF FALSE NEGATIVE ---")
print(false_negative.head())

print("\n --- FALSE POSITIVE ---")
print(false_positive.head())

report = classification_report(y_test_np, y_pred_np)
with open("output/evaluation_report.txt", "a") as f:
    f.write("\n====================\n")
    f.write("\n--- EVALUATION REPORT ---\n")
    f.write(report)
    f.write("\n--- CONFUSION MATRIX ---\n")
    f.write(str(confusion_matrix(y_test_np, y_pred_np)))

print("✅ Results saved in 'output/evaluation_report.txt'")