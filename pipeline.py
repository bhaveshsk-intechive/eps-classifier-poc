from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import re


def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    return text


# Expanded data: add these to your list
texts = [
    # True examples
    "validate the eps",
    "validate eps config",
    "check eps status",
    "run eps agent",
    "trigger eps workflow",
    # False examples
    "what is eps",
    "define eps",
    "explain eps",
    "eps meaning",
    # ... your original 10-20
]
labels = ["eps_agent_question"] * 5 + ["other"] * 4 + [...]  # Match lengths

texts = [preprocess(t) for t in texts]

vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", max_features=100, stop_words="english")
model = Pipeline([("tfidf", vectorizer), ("clf", LogisticRegression(C=0.01, penalty="l2", solver="liblinear", max_iter=200))])
model.fit(texts, labels)

# Test
print(model.predict_proba([preprocess("validate the eps")])[:, 1])  # Should >0.7
print(model.predict_proba([preprocess("what is eps")])[:, 1])  # Should <0.7

with open("improved_eps_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
