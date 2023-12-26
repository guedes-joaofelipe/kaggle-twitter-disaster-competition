import os

import dvc.api
import mlflow
import pandas as pd

from src import files
from src.models.utils import get_model
from src.utils.mlflow_run_decorator import mlflow_run


@mlflow_run
def train(filepath: str):
    df = files.load_dataset(filepath)

    params = dvc.api.params_show()
    import ipdb

    ipdb.set_trace()

    best_params, best_model = run_experiments(df, params["train"])


def run_experiments(df: pd.DataFrame, params: dict):
    best_params = {}
    best_model = None
    import ipdb

    ipdb.set_trace()
    Model = get_model(params["model_name"])
    model = Model(**params["model_params"])
    # model.fit()
    return best_params, best_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help="Path to file with training data")

    args = parser.parse_args()

    train(args.filepath)
