from typing import Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class Sbert(BaseEstimator, TransformerMixin):
    def __init__(self, pretrained_model="all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        self.model = SentenceTransformer(self.pretrained_model)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values.ravel()

        X = self.model.encode(X)

        return X
