from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter,
)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class SGD:
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
        config = {k: True if v == "True" else v for k, v in config.items()}
        config = {k: False if v == "False" else v for k, v in config.items()}
        # Add class weight (see https://github.com/mlr-org/mlr/blob/main/R/WeightedClassesWrapper.R)
        pw = self.config.get("pos_class_weight")

        config.update({'class_weight': {1: 2 ** pw, 0: 1}})
        del config['pos_class_weight']
        categorical_preproc = OneHotEncoder(handle_unknown="ignore")
        numerical_preproc = Pipeline([
            ("imputation",  SimpleImputer(strategy="median", add_indicator=True)),
            ("scaling", QuantileTransformer(output_distribution="normal"))
        ])
        column_transformer = ColumnTransformer([
            ("categorical", categorical_preproc, X.dtypes == "category"),
            ("numerical", numerical_preproc, X.dtypes != "category")
        ])

        estimator = SGDClassifier(random_state=self.seed, **config)
        pipeline = Pipeline([("preproc", column_transformer), ("estimator", estimator)])
        pipeline.fit(X, y, estimator__sample_weight=sample_weight)
        self.estimator = pipeline

    @classmethod
    def space(cls, seed: int | None = None) -> ConfigurationSpace:
        config_space = ConfigurationSpace(seed=seed)
        config_space.add_hyperparameters(
            [
                Constant("loss", "log_loss"),
                CategoricalHyperparameter("penalty", ["l2", "l1", "elasticnet"], default_value="l2"),
                UniformFloatHyperparameter("alpha", 1e-6, 1e-2, default_value=0.0001, log=True),
                UniformFloatHyperparameter("l1_ratio", 0., 1.0, default_value=0.15),
                CategoricalHyperparameter("fit_intercept", ["True", "False"], default_value="True"),
                UniformFloatHyperparameter("pos_class_weight", -7, 7, default_value=0),
                UnParametrizedHyperparameter("tol", 1e-3),
                UnParametrizedHyperparameter("shuffle", "True"),
                UnParametrizedHyperparameter("learning_rate", "adaptive"),
                UniformFloatHyperparameter("eta0", 1e-7, 1e-1, default_value=1e-3, log=True),
                UnParametrizedHyperparameter("early_stopping", "False"),
                UnParametrizedHyperparameter("validation_fraction", 0.1),
                UnParametrizedHyperparameter("n_iter_no_change", 5),
                UnParametrizedHyperparameter("warm_start", "False"),
                UnParametrizedHyperparameter("average", "False"),
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
    ) -> SGDClassifier:
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
