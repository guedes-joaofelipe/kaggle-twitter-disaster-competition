import os
from functools import wraps

import mlflow

from src.constants import PROJECT_EXPERIMENT_NAME, PROJECT_TRACKING_URI


def mlflow_run(wrapped_function):
    @wraps(wrapped_function)
    def wrapper(*args, **kwargs):
        mlflow.set_tracking_uri(PROJECT_TRACKING_URI)
        mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)
        with mlflow.start_run(run_id=os.environ["MLFLOW_PARENT_RUN_ID"]) as parent_run:
            with mlflow.start_run(
                run_name=wrapped_function.__name__,
                tags={"MLFLOW_PARENT_RUN_ID": os.environ["MLFLOW_PARENT_RUN_ID"]},
                nested=True,
            ) as child_run:
                return wrapped_function(*args, **kwargs)

    return wrapper
