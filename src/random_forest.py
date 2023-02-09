from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class RandomForest:
    def __init__(self, *, config: Configuration, seed: int | None = None):
        self.config = config
        self.seed = seed

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        config = {k: v if v != "None" else None for k, v in self.config.items()}
        # Add class weight (see https://github.com/mlr-org/mlr/blob/main/R/WeightedClassesWrapper.R)
        pw = self.config.get("pos_class_weight")

        config.update({'class_weight': {1: 2 ** pw, 0: 1}})
        del config['pos_class_weight']
        categorical_preproc = OneHotEncoder(handle_unknown="ignore")
        numerical_preproc = SimpleImputer(strategy="median", add_indicator=True)
        column_transformer = ColumnTransformer([
            ("categorical", categorical_preproc, X.dtypes == "category"),
            ("numerical", numerical_preproc, X.dtypes != "category")
        ])

        estimator = RandomForestClassifier(random_state=self.seed, **config)
        pipeline = Pipeline([("preproc", column_transformer), ("estimator", estimator)])
        pipeline.fit(X, y, estimator__sample_weight=sample_weight)
        self.estimator = pipeline

    @classmethod
    def space(cls, seed: int | None = None) -> ConfigurationSpace:
        config_space = ConfigurationSpace(seed=seed)
        config_space.add_hyperparameters(
            [
                CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini"),
                CategoricalHyperparameter("bootstrap", ["True", "False"], default_value="True"),
                UniformFloatHyperparameter("max_features", 0.0, 1.0, default_value=0.5),
                UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2),
                UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1),
                UniformFloatHyperparameter("pos_class_weight", -7, 7, default_value=0),
                UnParametrizedHyperparameter("max_depth", "None"),
                UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.0),
                UnParametrizedHyperparameter("max_leaf_nodes", "None"),
                UnParametrizedHyperparameter("min_impurity_decrease", 0.0),
            ]
        )
        return config_space

    @classmethod
    def random(
        cls,
        n: int | None = None,
        *,
        seed: int | None = None,
        estimator_kwargs: dict[str, Any] | None = None,
    ) -> RandomForest:
        """Generate a random estimator"""
        space = cls.space(seed=seed)
        if estimator_kwargs is None:
            estimator_kwargs = {}

        if n is None:
            config = space.sample_configuration()
            return cls(config=config, seed=seed, **estimator_kwargs)
        else:
            configs = space.sample_configuration(size=n)
            return [cls(config=config, seed=seed, **estimator_kwargs) for config in configs]
