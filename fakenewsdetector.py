import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# TF - times term appears in the article.

# IDF - Inverse document frequency a metric that is calculated with logarithm

# number of document / number of documents that contain the term

# multiply TF with IDF to get the score

data = pd.read_csv("fake_or_real_news.csv")
# print(data)

data['FAKE'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)

# print(data)

data = data.drop("label", axis=1)
# print(data)

X, y = data['text'], data['FAKE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

x_train_vectorizer = vectorizer.fit_transform(X_train)

x_test_vectorizer = vectorizer.transform(X_test)

classifier = LinearSVC()

classifier.fit(x_train_vectorizer, y_train)

print(classifier.score(x_test_vectorizer, y_test))





