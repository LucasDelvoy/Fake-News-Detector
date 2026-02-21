import re
import joblib
import numpy as np
from nltk.corpus import stopwords
from model import Model
from scipy.sparse import hstack
from torch import FloatTensor, no_grad, load
from vectorizer import text_subjectivity, title_subj, title_pol
from output.config import VEC_TEXT_PATH, VEC_TITLE_PATH, MODEL_PATH

model = Model()
stopwords_list = set(stopwords.words("english"))
vec_text = joblib.load(VEC_TEXT_PATH)
vec_title = joblib.load(VEC_TITLE_PATH)



def clean(text):
    txt = text.lower()
    x = re.sub(r'[^a-zA-Z]', " ", txt)
    words = x.split()

    cleaned_words = [word for word in words if word not in stopwords_list]
    cleaned_text = " ".join(cleaned_words)
    return cleaned_text



def predict(title, text):
    cleaned_text = clean(text)

    v_text = vec_text.transform([cleaned_text])
    v_title = vec_title.transform([title])

    text_subj = text_subjectivity(cleaned_text)
    title_subjectivity = title_subj(title)
    title_polarity = title_pol(title)

    ts = np.array([[text_subj]])
    tis = np.array([[title_subjectivity]])
    tip = np.array([[title_polarity]])

    article = FloatTensor(hstack([v_text, v_title, ts, tis, tip]).toarray())

    model.load_state_dict(load(MODEL_PATH, weights_only=True))
    model.eval()

    with no_grad():
        prediction = model(article)
        result = prediction.item()

        if result < 0.5:
            return "Fake news"
        else:
            return "Real news"



def main():
    title = input("Title: ")
    text = input("Text: ")

    result = predict(title, text)
    print(f"Result: {result}")



if __name__ == "__main__":
    main()