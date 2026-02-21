import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer


# Cleaning assets
stopwords_list = set(stopwords.words("english"))
vectorizer = TfidfVectorizer(max_features=20, stop_words="english")

def remove_stopwords(text):
    return " ".join([w for w in str(text).split() if w not in stopwords_list])

# Read csv file
df = pd.read_csv("news.csv")
df = df.sample(2000).reset_index(drop=True)

# Drop duplicates and empty entries
df = df.drop_duplicates()
df = df.dropna(subset=["text"])

# Get average number of words for real and fake news
index = df["label"].value_counts()

split_df = df["text"].str.split()
df["word_count"] = split_df.str.len()
df_average = df.groupby(["label"])["word_count"].mean()

# Get most used words in fake news
df["text"] = df["text"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
df["text"] = df["text"].apply(remove_stopwords)

# Save cleaned_data
df.to_csv("cleaned_data.csv", index=False)

fake_texts = df[df["label"] == 0]["text"]
fake_texts_matrix = vectorizer.fit_transform(fake_texts)
weights = fake_texts_matrix.toarray().mean(axis=0)
words = vectorizer.get_feature_names_out()

fake_dict = dict(zip(words, weights))
top_words_df = pd.DataFrame(list(fake_dict.items()), columns=['word', 'count']).sort_values(by='count', ascending=False)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(fake_dict)

# Visualisation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sns.histplot(data=df, x="word_count", hue="label", ax=ax1)
ax1.set_title("Word distribution")

sns.barplot(data=top_words_df, x="count", y="word", ax=ax2)
ax2.set_title("Most used words in fake news")

ax3.imshow(wordcloud, interpolation="bilinear")
ax3.axis("off")
ax3.set_title("Fake News wordcloud")

plt.tight_layout()
plt.show()