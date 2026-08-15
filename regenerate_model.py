import pickle
import string

import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score

for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(
            "tokenizers/" + resource
            if resource != "stopwords"
            else "corpora/" + resource
        )
    except LookupError:
        nltk.download(resource)

df = pd.read_csv("spam.csv", encoding="latin-1")

df.drop(
    columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
    inplace=True
)

df.rename(
    columns={"v1": "target", "v2": "text"},
    inplace=True
)

encoder = LabelEncoder()
df["target"] = encoder.fit_transform(df["target"])

df = df.drop_duplicates(keep="first")

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for token in text:
        if token.isalnum():
            y.append(token)

    text = y[:]
    y.clear()

    for token in text:
        if (
            token not in stopwords.words("english")
            and token not in string.punctuation
        ):
            y.append(token)

    text = y[:]
    y.clear()

    for token in text:
        y.append(ps.stem(token))

    return " ".join(y)


df["transformed_text"] = df["text"].apply(transform_text)

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(
    df["transformed_text"]
)

y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("model.pkl", "wb") as f:
    pickle.dump(mnb, f)

print("\nSuccessfully regenerated model.pkl and vectorizer.pkl")
