import os

import mlflow
import typer

from src.constants import (
    PROJECT_EXPERIMENT_NAME,
    PROJECT_ROOT_PATH,
    PROJECT_TRACKING_URI,
)


def start_pipeline():
    mlflow.set_tracking_uri(PROJECT_TRACKING_URI)
    mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_artifact(os.path.join(PROJECT_ROOT_PATH, "dvc.yaml"))
        cmd = f"export MLFLOW_PARENT_RUN_ID={mlflow.active_run().info.run_id} && dvc experiments run --name {mlflow.active_run().info.run_name}"
        os.system(cmd)


if __name__ == "__main__":
    typer.run(start_pipeline)