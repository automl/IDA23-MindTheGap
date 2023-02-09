from __future__ import annotations

from typing import Sequence

import pickle
from pathlib import Path
import zipfile

import numpy as np

from src.eval import Eval, Evaluations, Score
from src.metrics import Metric


class RandomizedTrainer:
    runsdir: Path = Path("runs")

    def __init__(
        self,
        *,
        model_class,
        id: str,
        seed: int = 1,
        n: int = 5,
    ):
        self.id = id
        self.seed = seed
        self.n = n
        self.model_class = model_class
        self._estimators: list | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._estimators = self.model_class.random(
            n=self.n,
            seed=self.seed,
        )
        for i, e in enumerate(self._estimators):
            print(f"Training configuration {i}")
            e.fit(X, y)

        print("Done fitting!")

    @property
    def estimators(self) -> list:
        assert self._estimators is not None
        return self._estimators

    @property
    def n_estimators(self) -> int:
        return len(self.estimators)

    def load(self) -> RandomizedTrainer:
        zipped_model = Path(str(self.modelpath) + '.zip')
        if zipped_model.exists():
            with zipfile.ZipFile(zipped_model) as f:
                filename = f.infolist()[0].filename
                model = pickle.loads(f.read(filename))

        with self.modelpath.open("rb") as f:
            model = pickle.load(f)
        print("Loaded")
        return model

    def save(self) -> None:
        path = self.modelpath
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        print(f"Saving to {self.modelpath}")

        with path.open("wb") as f:
            pickle.dump(self, f)

    @property
    def rundir(self) -> Path:
        return self.runsdir / str(self.id)

    @property
    def modelpath(self) -> Path:
        return self.rundir / "model.pkl"

    def evalpath(self, name: str) -> Path:
        return self.rundir / name

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        metrics: Sequence[Metric],
        name: str | None = None,
    ) -> Evaluations:
        # Load in any previously named data
        if name is not None and self.evalpath(name).exists():
            return self.load_evals(name)

        evals = [
            Eval(
                id=i,
                estimator=est,
                scores=[
                    Score(
                        value=metric(y_true=y, y_pred=est.predict(X)),
                        metric=metric.name,
                    )
                    for metric in metrics
                ],
            )
            for i, est in enumerate(self.estimators)
        ]
        evaluations = Evaluations(evals, [m.name for m in metrics])

        if name is not None:
            path = self.evalpath(name)
            if not path.parent.exists():
                path.parent.mkdir(exist_ok=True)

            with path.open("wb") as f:
                pickle.dump(evaluations, f)

            print(f"Saved evaluations for {name} to {path}")

        return evaluations

    def load_evals(self, name: str) -> Evaluations:
        evalpath = self.evalpath(name)
        zipped_evalpath = Path(str(self.evalpath(name)) + '.zip')
        
        if zipped_evalpath.exists():
            print(f"Loading evaluations for {name} from {zipped_evalpath}")
            with zipfile.ZipFile(zipped_evalpath) as f:
                filename = f.infolist()[0].filename
                return pickle.loads(f.read(filename))
        
        print(f"Loading evaluations for {name} from {evalpath}")
        with evalpath.open("rb") as f:
            return pickle.load(f)
