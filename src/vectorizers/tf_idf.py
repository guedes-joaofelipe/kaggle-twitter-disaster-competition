from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf(BaseEstimator, TransformerMixin):
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values.ravel()
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(X)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values.ravel()

        X = self.vectorizer.transform(X)

        return X.toarray()
