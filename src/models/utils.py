from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.random import RandomClassifier

MODELS = {
    "RandomClassifier": RandomClassifier,
    "LogisticRegressionClassifier": LogisticRegressionClassifier,
}


def get_model(model_name: str):
    if model_name not in MODELS.keys():
        raise NotImplementedError("Model is not implemented")

    return MODELS[model_name]
