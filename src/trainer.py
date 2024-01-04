import hyperopt as hopt
import ipdb
import mlflow
import numpy as np
from hyperopt import hp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.utils import get_model
from src.vectorizers.utils import get_vectorizer


class Trainer:
    def __init__(self, params: dict) -> None:
        self.params = params

        self.model_params = params["train"]["model"]
        self.preprocessor_params = params["train"]["preprocessor"]
        self.hyperopt_params = params["train"]["hyperopt"]
        self.metrics_params = params["metrics"]
        self.space = self.render_params()

    def render_params(self) -> dict:
        params = {}
        space = self.hyperopt_params["space"]
        for param, values in space.items():
            exec(f"params['{param}'] = {values}")

        return {"params": params}

    def run_experiments(self, X, y) -> dict:
        trials = hopt.Trials()
        self.num_features = X.select_dtypes(include="number").columns.to_list()
        self.text_feature = "text_clean"

        best_model_params = hopt.fmin(
            fn=lambda args: self.objective(args, X, y),
            space=self.space,
            algo=hopt.tpe.suggest,
            max_evals=self.hyperopt_params["max_evals"],
            trials=trials,
            verbose=True,
        )

        self.trials = trials.trials
        self.best_params = hopt.space_eval(self.space, best_model_params)

        pipeline = self.get_pipeline()
        pipeline.set_params(**self.best_params["params"])
        pipeline.fit(X, y)

        self.best_pipeline = pipeline
        self.best_metrics = trials.best_trial["result"]
        self.best_metrics.pop("status")

        experiments_results = {
            "best_pipeline": self.best_pipeline,
            "best_params": self.best_params["params"],
            "best_metrics": self.best_metrics,
        }

        return experiments_results

    def objective(self, args, X, y):
        pipeline = self.get_pipeline()
        pipeline.set_params(**args["params"])

        scoring = [self.metrics_params["loss"]] + self.metrics_params["auxiliary"]

        scores = cross_validate(
            pipeline,
            X,
            y,
            scoring=scoring,
            cv=self.hyperopt_params["cross_validation"],
            n_jobs=self.hyperopt_params["n_jobs"],
            return_train_score=True,
            error_score=0.99,
        )

        results = {
            "loss": 1 - np.mean(scores["test_" + self.metrics_params["loss"]]),
        }

        for metric_name, metric_values in scores.items():
            results[metric_name] = np.mean(metric_values)

        results["status"] = hopt.STATUS_OK

        return results

    def get_pipeline(self) -> Pipeline:
        Vectorizer = get_vectorizer(self.preprocessor_params["vectorizer_name"])
        Model = get_model(self.model_params["name"])

        text_transformer = Vectorizer()
        num_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ("text_transformer", text_transformer, [self.text_feature]),
                ("num_transformer", num_transformer, self.num_features),
            ]
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Model()),
            ]
        )
