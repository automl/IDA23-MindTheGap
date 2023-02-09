from __future__ import annotations

from typing import Callable, Union

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

MetricSig = Callable[[np.ndarray, np.ndarray, Union[np.ndarray, None]], float]


@dataclass
class Metric:
    name: str
    f: MetricSig = field(repr=False, compare=False)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.f(y_true, y_pred)


def _classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - accuracy_score(y_true, y_pred)


def _precision_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - precision_score(y_true, y_pred)


def _recall_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1 - recall_score(y_true, y_pred)


classification_error = Metric("accuracy_error", f=_classification_error)
precision_error = Metric("precision_error", f=_precision_error)
recall_error = Metric("recall_error", f=_recall_error)


if __name__ == "__main__":
    for m in [classification_error, precision_error, recall_error]:
        print(m)
