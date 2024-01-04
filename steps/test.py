import os

import dvc.api
import mlflow
import pandas as pd
import typer

from src import files
from src.decorators import mlflow_run


@mlflow_run
def test(filepath: str):
    params = dvc.api.params_show()
    df = files.load_dataset(filepath)
    df = df[params["features"]]

    mlflow.log_input(mlflow.data.from_pandas(df, source=filepath), context="test")

    active_run = mlflow.active_run()

    model_uri = os.path.join(
        active_run.info.artifact_uri, params["train"]["model"]["name"]
    )

    model = mlflow.sklearn.load_model(model_uri)
    df_predictions = pd.DataFrame(index=df.index)
    df_predictions["target"] = model.predict(df)
    predictions_filepath = os.path.join(
        params["test"]["path"], f"{active_run.info.run_name}.csv"
    )
    files.save_dataset(df_predictions, predictions_filepath)
    mlflow.log_artifact(predictions_filepath, artifact_path="predictions")


if __name__ == "__main__":
    typer.run(test)
