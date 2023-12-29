import os

import kaggle
import mlflow
import pandas as pd
import typer
from mlflow.entities.run import Run

from src import files
from src.constants import (
    PROJECT_COMPETITION,
    PROJECT_EXPERIMENT_NAME,
    PROJECT_ROOT_PATH,
    PROJECT_TRACKING_URI,
)


def submit(model_name: str, version: str):
    mlflow.set_tracking_uri(PROJECT_TRACKING_URI)
    mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)
    import ipdb

    registered_models = mlflow.search_registered_models(
        filter_string=f"name = '{model_name}'"
    )

    if len(registered_models) == 0:
        raise ValueError("Model not found")

    registered_model = registered_models[0]
    model = None
    for model_version in registered_model.latest_versions:
        if model_version.version == version:
            model = model_version

    if model is None:
        raise ValueError("Model version not found")

    parent_run = mlflow.get_run(model.run_id)

    submission_filename = f"{parent_run.info.run_name}.csv"

    df = get_submissions(submission_filename)

    if df.shape[0] > 0:
        print(df.head())
        force_submission = input(
            f"Submission {submission_filename} already exists. Force submission: "
        )

    if df.shape[0] == 0 or force_submission.startswith("y"):
        if force_submission:
            print("Forcing submission")
        test_run = get_test_run(parent_run)
        predictions_filepath = os.path.join(
            test_run.info.artifact_uri, "predictions", submission_filename
        )

        message = f"""
        Model: {model.name} (v{model.version})
        Description: {model.description}
        """

        kaggle.api.competition_submit(
            file_name=predictions_filepath,
            competition=PROJECT_COMPETITION,
            message=message,
        )
        df = get_submissions(submission_filename)
    else:
        print(f"Skipping submission")

    output_filepath = f"data/submit/{submission_filename}"
    files.save_dataset(df, output_filepath)
    with mlflow.start_run(run_id=parent_run.info.run_id):
        mlflow.log_artifact(local_path=output_filepath, artifact_path="submission")
        mlflow.log_metric("public_score", df["publicScore"].astype(float).values[0])


def get_submissions(filename: str) -> pd.DataFrame:
    submissions = kaggle.api.competitions_submissions_list(id=PROJECT_COMPETITION)
    df = pd.DataFrame.from_dict(submissions)
    df = df[(df["fileName"] == filename) & (df["publicScore"] != "")].sort_values(
        by="publicScore", ascending=False
    )
    df["publicScore"] = df["publicScore"].astype(float)
    return df


def get_test_run(parent_run: Run) -> Run:
    df = mlflow.search_runs(experiment_ids=[parent_run.info.experiment_id])
    df = df[
        (df["tags.mlflow.parentRunId"] == parent_run.info.run_id)
        & (df["tags.mlflow.runName"] == "test")
    ]

    test_run_id = df["run_id"].values[0]
    return mlflow.get_run(test_run_id)


if __name__ == "__main__":
    typer.run(submit)
