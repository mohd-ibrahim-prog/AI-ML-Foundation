"""
model_trainer.py

Purpose:
    Build and train a TF-IDF + Multinomial Naive Bayes spam
    classification pipeline.

Week:
    AI/ML Foundation - Week 3
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train_model(X_train, y_train) -> Pipeline:
    """
    Train a TF-IDF + Multinomial Naive Bayes pipeline on the training data.

    Bundling the TF-IDF vectorizer and the classifier into a single
    scikit-learn Pipeline means the exact same vectorizer (fitted on
    the training vocabulary) is always used at prediction time.

    Args:
        X_train: Training messages (already cleaned text).
        y_train: Training labels ('ham' / 'spam').

    Returns:
        Pipeline: Fitted TF-IDF + Naive Bayes pipeline.
    """

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("naive_bayes", MultinomialNB()),
    ])

    model.fit(X_train, y_train)

    print("\nTF-IDF + Multinomial Naive Bayes model trained successfully.")

    vocabulary_size = len(model.named_steps["tfidf"].vocabulary_)
    print(f"TF-IDF vocabulary size : {vocabulary_size}")

    return model
