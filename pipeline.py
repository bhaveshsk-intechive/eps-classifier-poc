from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Example data
texts = [
    "trigger eps workflow for this account",
    "can you run eps agent on customer 123?",
    "what is the weather today",
    "book a cab for me",
    # ... add 10-20 questions total, with labels
]

labels = [
    "eps_agent_question",
    "eps_agent_question",
    "other",
    "other",
    # ...
]

# Pipeline: vectorizer + classifier
model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=300, stop_words="english")),
        ("clf", LogisticRegression(C=0.1, penalty="l2", solver="liblinear", max_iter=1000)),
    ]
)

model.fit(texts, labels)

# Save to pickle
with open("eps_agent_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
