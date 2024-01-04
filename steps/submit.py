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
    mlflow.set_experiment(model_name)
    import ipdb

    model_versions = mlflow.search_model_versions(
        filter_string=f"name = '{model_name}'", order_by={"version_number": "ASC"}
    )

    if len(model_versions) == 0:
        raise ValueError("Model not found")

    model = None
    if int(version) == -1:
        model = model_versions[-1]
        print("Latest version loaded")
    else:
        for model_version in model_versions:
            if model_version.version == version:
                model = model_version
                print(f"Version {version} loaded")

    if model is None:
        raise ValueError("Model version not found")

    ipdb.set_trace()
    parent_run = mlflow.get_run(model.run_id)

    submission_filename = f"{parent_run.info.run_name}.csv"
    # submission_filename = "hilarious-ram-130.csv"

    df = get_submissions(submission_filename)

    force_submission = "No"
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

    import ipdb

    ipdb.set_trace()

    df = df[["fileName", "description", "publicScore"]].head(1)
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
    df = df[(df["run_id"] == parent_run.info.run_id)]

    test_run_id = df["run_id"].values[0]
    return mlflow.get_run(test_run_id)


if __name__ == "__main__":
    typer.run(submit)
