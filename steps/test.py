import json
import os

import dvc.api
import mlflow
import pandas as pd
import typer
from mlflow import MlflowClient
from mlflow.entities.model_registry import RegisteredModel
from mlflow.entities.run import Run

from src import files
from src.models.utils import get_model
from src.utils.mlflow_run_decorator import mlflow_run


@mlflow_run
def test(filepath: str):
    df = files.load_dataset(filepath)

    active_run = mlflow.active_run()
    parent_run = mlflow.get_parent_run(active_run.info.run_id)
    params = dvc.api.params_show()

    model_uri = os.path.join(
        parent_run.info.artifact_uri, params["train"]["model"]["name"]
    )

    model = mlflow.sklearn.load_model(model_uri)
    df_predictions = pd.DataFrame(index=df.index)
    df_predictions["predictions"] = model.predict(df)
    predictions_filepath = params["test"]["predictions_path"]
    files.save_dataset(df_predictions, predictions_filepath)
    mlflow.log_artifact(predictions_filepath, artifact_path="predictions")


if __name__ == "__main__":
    typer.run(test)
