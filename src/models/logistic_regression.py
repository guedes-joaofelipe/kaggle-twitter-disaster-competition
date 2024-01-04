from typing import Literal, Mapping, Union

import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier(LogisticRegression):
    name = "LogisticRegressionClassifier"
    description = "SKLearn Logistic Regression"
    tags = {"framework": "sklearn"}

    def __init__(
        self,
        penalty: Union[Literal["l1", "l2", "elasticnet"], None] = "l2",
        *,
        dual: bool = False,
        tol: float = 0.0001,
        C: float = 1,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Union[Mapping, str, None] = None,
        random_state: Union[int, RandomState, None] = None,
        solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = "lbfgs",
        max_iter: int = 100,
        multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Union[int, None] = None,
        l1_ratio: Union[float, None] = None
    ) -> None:
        super().__init__(
            penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )
