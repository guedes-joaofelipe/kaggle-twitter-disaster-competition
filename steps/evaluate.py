import os

import dvc.api
import mlflow
import pandas as pd
import typer

from src import files
from src.decorators import mlflow_run


@mlflow_run
def evaluate(filepath: str):
    params = dvc.api.params_show()
    df = files.load_dataset(filepath)
    df = df[[params["target"]] + params["features"]]

    active_run = mlflow.active_run()

    model_uri = os.path.join(
        active_run.info.artifact_uri, params["train"]["model"]["name"]
    )

    model = mlflow.sklearn.load_model(model_uri)

    df["prediction"] = model.predict(df[params["features"]])

    def fn(X):
        return model.predict(X)

    evaluation = mlflow.evaluate(
        data=df,
        model=None,
        targets="target",
        predictions="prediction",
        model_type="classifier",
    )


if __name__ == "__main__":
    typer.run(evaluate)
