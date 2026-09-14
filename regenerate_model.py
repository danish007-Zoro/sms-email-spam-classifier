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


# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# --------------------------------------------------
# 1. Load and clean dataset
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------

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


# --------------------------------------------------
# 3. TF-IDF feature extraction
# --------------------------------------------------

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(
    df["transformed_text"]
).toarray()

y = df["target"].values


# --------------------------------------------------
# 4. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)


# --------------------------------------------------
# 5. Train Multinomial Naive Bayes
# --------------------------------------------------

mnb = MultinomialNB()

mnb.fit(X_train, y_train)


# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------

y_pred = mnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")


# --------------------------------------------------
# 7. Save fitted artifacts
# --------------------------------------------------

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("model.pkl", "wb") as f:
    pickle.dump(mnb, f)


print("\nSuccessfully regenerated:")
print("  vectorizer.pkl")
print("  model.pkl")