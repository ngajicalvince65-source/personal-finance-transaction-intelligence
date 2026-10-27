"""Custom model classes used by the deployed classifier.

The Part 3 notebook defined TextGatedHybrid inside the notebook itself, so the saved model
file referred to it as __main__.TextGatedHybrid. Any other program (the API, the Streamlit app,
a container) cannot load such a file. Defining the class here, in an importable module,
fixes that permanently.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

TEXT_COL = "description_clean"


class TextGatedHybrid(BaseEstimator, ClassifierMixin):
    """Uses the text model when the description contains words seen in training,
    otherwise uses the behavior model. Probabilities come from whichever model is used."""

    def __init__(self, text_model=None, behavior_model=None):
        self.text_model = text_model
        self.behavior_model = behavior_model

    def fit(self, X, y):
        self.text_model_ = clone(self.text_model).fit(X, y)
        self.behavior_model_ = clone(self.behavior_model).fit(X, y)
        self.classes_ = self.text_model_.classes_
        return self

    def known_words(self, X):
        vec = self.text_model_.named_steps["prep"].named_transformers_["text"]
        return np.asarray(vec.transform(X[TEXT_COL]).sum(axis=1)).ravel() > 0

    def predict_proba(self, X):
        known = self.known_words(X)[:, None]
        return np.where(known, self.text_model_.predict_proba(X), self.behavior_model_.predict_proba(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
