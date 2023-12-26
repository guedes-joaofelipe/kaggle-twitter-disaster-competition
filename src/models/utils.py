from src.models.random import RandomClassifier

MODELS = {"RandomClassifier": RandomClassifier}


def get_model(model_name: str):
    if model_name not in MODELS.keys():
        raise NotImplementedError("Model is not implemented")

    return MODELS[model_name]
