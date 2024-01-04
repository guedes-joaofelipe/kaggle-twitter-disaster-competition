import os

import dvc.api
import ipdb
import mlflow
import pandas as pd
import typer
from mlflow import MlflowClient
from mlflow.entities.model_registry import RegisteredModel
from sklearn.pipeline import Pipeline

from src import files
from src.decorators import mlflow_run
from src.trainer import Trainer


@mlflow_run
def train(filepath: str):
    params = dvc.api.params_show()

    df = files.load_dataset(filepath)
    df = df[[params["target"]] + params["features"]]

    mlflow.log_input(mlflow.data.from_pandas(df, source=filepath), context="train")
    y = df[params["target"]]
    X = df[params["features"]]

    # ipdb.set_trace()
    trainer = Trainer(params)
    experiments_results = trainer.run_experiments(X, y)

    # ipdb.set_trace()

    for trial_index, trial in enumerate(trainer.trials):
        for metric_name, metric_value in trial["result"].items():
            if metric_name != "status":
                mlflow.log_metric(metric_name, metric_value, step=trial_index + 1)

    mlflow.log_params(experiments_results["best_params"])
    mlflow.log_metrics(experiments_results["best_metrics"])

    register_pipeline(experiments_results["best_pipeline"], params, X.head(), y.head())


def register_pipeline(
    pipeline: Pipeline,
    params: dict,
    X,
    y,
    verbose: bool = True,
) -> RegisteredModel:
    model_name = pipeline.named_steps["model"].name
    client = MlflowClient()
    try:
        client.create_registered_model(
            name=model_name,
            tags=pipeline.named_steps["model"].tags,
            description=pipeline.named_steps["model"].description,
        )

    except Exception as e:
        print("Exception:", str(e))

    active_run = mlflow.active_run()

    output_folder = os.path.join(active_run.info.artifact_uri, model_name)
    files.remove_dir(output_folder)
    signature = mlflow.models.infer_signature(X, y)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=params["train"]["model"]["name"],
        signature=signature,
        input_example=X.head(),
        pyfunc_predict_fn="predict",
        pip_requirements="requirements.txt",
    )

    model_info = client.create_model_version(
        name=model_name,
        source=output_folder,
        run_id=active_run.info.run_id,
        tags=params["train"]["version"]["tags"],
        description=params["train"]["version"]["description"],
    )

    print(f"Created version v{model_info.version} for model {model_info.name}")
    os.makedirs("data/train", exist_ok=True)


if __name__ == "__main__":
    typer.run(train)
