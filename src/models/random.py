import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5):
        prediction_scores = self.predict_proba(X)
        return (prediction_scores >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray, y: np.ndarray = None):
        return np.random.random(X.shape[0], random_state=self.random_state)
