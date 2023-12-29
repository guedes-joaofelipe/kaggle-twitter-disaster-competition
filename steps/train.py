import os

import dvc.api
import hyperopt as hopt
import mlflow
import numpy as np
import pandas as pd
import typer
from mlflow import MlflowClient
from mlflow.entities.model_registry import RegisteredModel
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src import files
from src.models.utils import get_model
from src.utils.mlflow_run_decorator import mlflow_run


@mlflow_run
def train(filepath: str):
    params = dvc.api.params_show()

    df = files.load_dataset(filepath)
    mlflow.log_input(mlflow.data.from_pandas(df, source=filepath), context="train")

    y = df[params["target"]]
    X = df.drop(columns=params["target"])

    experiments_results = run_experiments(df, params, X, y)

    # ver a funcao mlflow.evaluate

    mlflow.log_params(experiments_results["best_params"]["params"])
    mlflow.log_params(experiments_results["best_params"]["hyperopt"])
    mlflow.log_metrics(experiments_results["best_metrics"])

    register_model(experiments_results["best_model"], params, X.head(), y.head())


def run_experiments(df: pd.DataFrame, params: dict, X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    Model = get_model(params["train"]["model"]["name"])
    space = {
        "model": Model(),
        "params": {"model__threshold": hopt.hp.uniform("threshold", 0.5, 0.6)},
    }

    trials = hopt.Trials()

    best_model_params = hopt.fmin(
        lambda x: objective(
            x,
            X_train,
            y_train,
            metrics=params["metrics"],
            cv=params["train"]["hyperopt"]["cross_validation"],
            n_jobs=params["train"]["hyperopt"]["n_jobs"],
        ),
        space,
        algo=hopt.tpe.suggest,
        max_evals=params["train"]["hyperopt"]["max_evals"],
        trials=trials,
        verbose=False,
    )
    for trial_index, trial in enumerate(trials.trials):
        for metric_name, metric_value in trial["result"].items():
            if metric_name != "status":
                mlflow.log_metric(metric_name, metric_value, step=trial_index + 1)

    best_params = hopt.space_eval(space, best_model_params)
    best_params["hyperopt"] = params["train"]["hyperopt"]

    best_model = best_params.pop("model")
    best_metrics = trials.best_trial["result"]
    best_metrics.pop("status")

    experiments_results = {
        "best_model": best_model,
        "best_params": best_params,
        "best_metrics": best_metrics,
    }

    return experiments_results


def objective(args, X, y, metrics: dict, cv: int = 3, n_jobs: int = -1):
    pipeline = Pipeline(
        steps=[
            (
                "model",
                args["model"],
            )
        ]
    )

    pipeline.set_params(**args["params"])

    scoring = [metrics["loss"]] + metrics["auxiliary"]

    scores = cross_validate(
        pipeline,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score=0.99,
    )

    results = {
        "loss": 1 - np.mean(scores["test_" + metrics["loss"]]),
        "status": hopt.STATUS_OK,
    }

    for metric_name, metric_values in scores.items():
        results[metric_name] = np.mean(metric_values)

    return results


def register_model(
    model: BaseEstimator,
    params: dict,
    X,
    y,
    verbose: bool = True,
) -> RegisteredModel:
    client = MlflowClient()
    try:
        client.create_registered_model(
            name=model.name,
            tags=model.tags,
            description=model.description,
        )

    except Exception as e:
        print("Exception:", str(e))

    active_run = mlflow.active_run()
    parent_run = mlflow.get_parent_run(active_run.info.run_id)
    output_folder = os.path.join(
        parent_run.info.artifact_uri, params["train"]["model"]["name"]
    )
    files.remove_dir(output_folder)
    signature = mlflow.models.infer_signature(X, y)

    mlflow.sklearn.save_model(
        sk_model=model,
        # artifact_path=params["train"]["model"]["name"],
        path=output_folder,
        signature=signature,
        input_example=X.head(),
        pyfunc_predict_fn="predict",
        pip_requirements="requirements.txt",
    )

    model_info = client.create_model_version(
        name=model.name,
        source=output_folder,
        run_id=parent_run.info.run_id,
        tags=params["train"]["version"]["tags"],
        description=params["train"]["version"]["description"],
    )

    print(f"Created version v{model_info.version} for model {model_info.name}")
    os.makedirs("data/train", exist_ok=True)


# adsad

if __name__ == "__main__":
    typer.run(train)


# kaggle competitions submit -c nlp-getting-started -f submission.csv -m "Message"
