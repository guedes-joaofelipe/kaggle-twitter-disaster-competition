from typing import Literal, Mapping, Sequence, Union

from numpy.random import RandomState
from sklearn.ensemble import RandomForestClassifier as RandomForest


class RandomForestClassifier(RandomForest):
    name = "RandomForestClassifier"
    description = "SKLearn Random Forest"
    tags = {"framework": "sklearn"}

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        max_depth: Union[int, None] = None,
        min_samples_split: Union[float, int] = 2,
        min_samples_leaf: Union[float, int] = 1,
        min_weight_fraction_leaf: float = 0,
        max_features: Union[float, int, Literal["sqrt", "log2"]] = "sqrt",
        max_leaf_nodes: Union[int, None] = None,
        min_impurity_decrease: float = 0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Union[int, None] = None,
        random_state: Union[int, RandomState, None] = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Union[
            Mapping, Sequence[Mapping], Literal["balanced", "balanced_subsample"], None
        ] = None,
        ccp_alpha: float = 0,
        max_samples: Union[float, int, None] = None
    ) -> None:
        super().__init__(
            n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
