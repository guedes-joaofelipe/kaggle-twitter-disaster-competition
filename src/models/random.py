import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomClassifier(BaseEstimator, ClassifierMixin):
    name = "RandomClassifier"
    description = "Generates a random number on prediction"
    tags = {"framework": "sklearn"}

    def __init__(self, random_state: int = 0, threshold: float = 0.5):
        self.random_state = random_state
        self.threshold = threshold
        np.random.seed(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self

    def predict(self, X: np.ndarray):
        prediction_scores = self.predict_proba(X)
        return (prediction_scores >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray, y: np.ndarray = None):
        return np.random.random(X.shape[0])
