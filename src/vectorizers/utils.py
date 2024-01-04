from sklearn.feature_extraction.text import CountVectorizer

from src.vectorizers.sbert import Sbert
from src.vectorizers.tf_idf import TfIdf

VECTORIZERS = {"tfidf": TfIdf, "sbert": Sbert}


def get_vectorizer(vectorizer_name: str) -> CountVectorizer:
    if vectorizer_name not in VECTORIZERS.keys():
        raise NotImplementedError("Vectorizer not implemented")

    return VECTORIZERS[vectorizer_name]
