import pandas as pd
import joblib
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Functions
def text_subjectivity(text):
    zen = TextBlob(text)
    subjectivity = zen.sentiment.subjectivity
    return subjectivity

def title_subj(title):
    zen = TextBlob(title)
    subjectivity = zen.sentiment.subjectivity
    return subjectivity

def title_pol(title):
    zen = TextBlob(title)
    polarity = zen.sentiment.polarity
    return polarity

def main():
    # Prepare assets
    df = pd.read_csv("datasets/cleaned_data.csv")
    df = df.dropna(subset=["text"])

    # Split dataset
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Vectorizers
    vec_text = TfidfVectorizer(sublinear_tf=True,
                            max_features=10000,
                            ngram_range=(1, 2))

    vec_title = TfidfVectorizer(sublinear_tf=True,
                                max_features=1000,
                                lowercase=False)

    # Put dataset through TextBlob
    df_train["subj_score"] = df_train["text"].apply(text_subjectivity)
    df_test["subj_score"] = df_test["text"].apply(text_subjectivity)

    df_train["title_subj_score"] = df_train["title"].apply(title_subj)
    df_train["title_pol_score"] = df_train["title"].apply(title_pol)
    df_test["title_subj_score"] = df_test["title"].apply(title_subj)
    df_test["title_pol_score"] = df_test["title"].apply(title_pol)

    # Reshape TextBlob
    X_train_subj_text = df_train["subj_score"].values.reshape(-1, 1)
    X_train_subj_title = df_train["title_subj_score"].values.reshape(-1, 1)
    X_train_pol_title = df_train["title_pol_score"].values.reshape(-1, 1)

    X_test_subj_text = df_test["subj_score"].values.reshape(-1, 1)
    X_test_subj_title = df_test["title_subj_score"].values.reshape(-1, 1)
    X_test_pol_title = df_test["title_pol_score"].values.reshape(-1, 1)

    # Put dataset through vectorizers
    X_text_train = vec_text.fit_transform(df_train["text"])
    X_text_test = vec_text.transform(df_test["text"])

    X_title_train = vec_title.fit_transform(df_train["title"])
    X_title_test = vec_title.transform(df_test["title"])

    # Fuse X_text and X_title
    X_train = hstack([X_text_train, X_title_train, X_train_subj_text, X_train_subj_title, X_train_pol_title])
    X_test = hstack([X_text_test, X_title_test, X_test_subj_text, X_test_subj_title, X_test_pol_title])

    # Get labels
    y_train = df_train["label"]
    y_test = df_test["label"]

    # Print shape
    print(f"{X_train.shape}, {X_test.shape}")

    data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    joblib.dump(data, "output/data.pkl")
    joblib.dump(vec_text, "output/vec_text.pkl")
    joblib.dump(vec_title, "output/vec_title.pkl")

if __name__ == "__main__":
    main()