import pandas as pd
import joblib
import torch
import numpy
from model import Model
from vectorizer import text_subjectivity, title_subj, title_pol
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics import classification_report, confusion_matrix

# import dataset
df = pd.read_csv("./datasets/cleaned_data.csv")

#import assets
model = Model()
data = joblib.load("./output/data.pkl")
vec_text = joblib.load("./output/vec_text.pkl")
vec_title = joblib.load("./output/vec_title.pkl")

# Erase labels and empty entries
df = df.dropna(subset=["text"])
y_true = df["label"]
df_features = df.drop(columns=["label"])

# Add subjectivity and polarity score
df_features["subj_score"] = df_features["text"].apply(text_subjectivity)
df_features["title_subj_score"] = df_features["title"].apply(title_subj)
df_features["title_pol_score"] = df_features["title"].apply(title_pol)

X_text_subj = df_features["subj_score"].values.reshape(-1, 1)
X_title_subj = df_features["title_subj_score"].values.reshape(-1, 1)
X_title_pol = df_features["title_pol_score"].values.reshape(-1, 1)

# Vectorize dataset
X_text = vec_text.transform(df_features["text"])
X_title = vec_title.transform(df_features["title"])

# Fuse vectors
X = hstack([X_text, X_title, X_text_subj, X_title_subj, X_title_pol])
X_array = X.toarray()
X_test = torch.FloatTensor(X_array)

# Load model
model.load_state_dict(torch.load("output/model.pth"))
model.eval()

with torch.no_grad():
    prediction = model(X_test)
    y_test = (prediction > 0.5).int()

y_test_np = y_test.numpy().flatten()
y_true_np = y_true.values.flatten()

print(classification_report(y_true, y_test_np))
print(confusion_matrix(y_true, y_test_np))

preds = y_test_np.tolist()
reels = y_true.values.tolist()

with open("output/evaluation_report.txt", "a") as f:
    f.write("\n--- ERROR ANALYSIS ---\n")
    for i in range(len(preds)):
        if preds[i] != reels[i]:
            f.write(f"CSV Index : {df.index[i]}\n")
            f.write(f"Title : {df.iloc[i]['title']}\n")
            f.write(f"True label : {reels[i]} | Predicted : {preds[i]}\n")
            f.write("-" * 30 + "\n")