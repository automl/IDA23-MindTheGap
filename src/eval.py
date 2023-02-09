from __future__ import annotations

from typing import (
    Any,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    MutableMapping,
    Sequence,
    TypeVar,
)

from dataclasses import dataclass, field
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.random_forest import RandomForest
from src.util import pareto_front

from src.deap_pyhv import hypervolume

ID = TypeVar("ID", bound=Hashable)


@dataclass(order=True, frozen=True)
class Score:
    value: float
    metric: str

    @property
    def name(self) -> str:
        return self.metric

    def __repr__(self) -> str:
        return f"Score({self.name}, {self.value})"


@dataclass(frozen=True)
class Eval(Generic[ID], Iterable[Score]):
    id: ID
    estimator: RandomForest = field(hash=False, repr=False, compare=False)
    scores: Sequence[Score] = field(hash=False)

    @property
    def point(self) -> tuple[float, ...]:
        return tuple(score.value for score in self.scores)

    def __iter__(self) -> Iterator[Score]:
        return iter(self.scores)


@dataclass(init=True)
class Evaluations(MutableMapping[ID, Eval[ID]]):
    evaluations: list[Eval[ID]]
    metrics: Sequence[str]

    # Used to allow fast lookup
    _map: dict[ID, Eval[ID]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._map = {e.id: e for e in self.evaluations}

    @cached_property
    def df(self) -> pd.DataFrame:
        ids, evals = zip(*self._map.items())
        return pd.DataFrame(
            data=[e.point for e in evals],
            columns=self.metrics,
            index=ids,
        )

    def __contains__(self, id: object) -> bool:
        return id in self._map

    def __getitem__(self, id: ID) -> Eval:
        return self._map[id]

    def only(self, ids: Iterable[ID]) -> Evaluations:
        return Evaluations([self._map[i] for i in ids], self.metrics)

    def __len__(self) -> int:
        return len(self.evaluations)

    def __iter__(self) -> Iterator[ID]:
        return iter(self._map)

    def pareto_front(self, optimistic: bool = True) -> Pareto[ID]:
        # Get the costs of each point
        costs = np.asarray([list(e.point) for e in self.evaluations])
        front = pareto_front(costs, optimistic=optimistic)

        pareto = [e for e, in_pareto in zip(self.evaluations, front) if in_pareto]
        return Pareto(pareto, self.metrics)

    def plot(self, *, ax: Axes | None = None, **kwargs: Any) -> Axes:
        assert len(self.metrics) == 2

        if ax is None:
            ax = plt.gca()

        if "marker" not in kwargs:
            kwargs["marker"] = "x"

        x, y = (self.df[self.metrics[0]], self.df[self.metrics[1]])
        ax.scatter(x, y, **kwargs)

        return ax

    def plot_arrows(
        self,
        to: Evaluations | None = None,
        frm: Evaluations | None = None,
        *,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        if (not to and not frm) or (to and frm):
            raise ValueError("Must specify (only) one of `to` or `frm`")

        other = to if to else frm
        assert other is not None

        missing = self.keys() - other.keys()
        if any(missing):
            raise ValueError(f"Other does not contain keys present in self\n{missing}")

        assert len(self.metrics) == 2

        if ax is None:
            ax = plt.gca()

        default = {
            "color": "grey",
            "width": 0.01,
        }
        kwargs = {**default, **kwargs}

        ids = self._map.keys()

        # Get our points and their points
        ours = [self[id].point for id in ids]
        theirs = [other[id].point for id in ids]

        # Define a list of segments going from one point to the other
        if to:
            dx, dy = zip(*[(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(ours, theirs)])
            xs, ys = zip(*ours)
        else:
            dx, dy = zip(*[(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(theirs, ours)])
            xs, ys = zip(*theirs)

        ax.quiver(xs, ys, dx, dy, angles="xy", scale_units="xy", scale=1, **kwargs)

        return ax

    def __delitem__(self, id: ID) -> None:
        eval = self._map[id]
        self.evaluations.remove(eval)
        del self._map[id]

    def __setitem__(self, id: ID, eval: Eval[ID]) -> None:
        if id in self._map:
            old_eval = self._map[id]
            idx = self.evaluations.index(old_eval)
            self.evaluations[idx] = eval

        self._map[id] = eval


class Pareto(Evaluations[ID]):
    def plot(
        self,
        *,
        ax: Axes | None = None,
        kind: str | list[str] = "pareto",
        endpoints: bool = True,
        **kwargs: Any,
    ) -> Axes:
        assert len(self.metrics) == 2

        if isinstance(kind, str):
            kind = [kind]

        assert all(k in ["pareto", "scatter"] for k in kind)

        if "scatter" in kind:
            super().plot(ax=ax, **kwargs)

        if "pareto" in kind:
            if ax is None:
                ax = plt.gca()

            if "marker" not in kwargs:
                kwargs["marker"] = "o"

            # Sort all the points and then unpack them into the x and y components
            sorted_points = sorted([e.point for e in self.evaluations])
            xs, ys = zip(*sorted_points)

            if endpoints:
                new_xs = [0] + list(xs) + [1]
                new_ys = [1] + list(ys) + [0]
            else:
                new_xs = list(xs)
                new_ys = list(ys)

            hv = self.hypervolume()
            if "label" in kwargs:
                kwargs["label"] += f" - {hv:.4f}"

            # line plots don't allow these
            for k in ["facecolor", "edgecolor"]:
                if k in kwargs:
                    del kwargs[k]

            for k, k_new in [("s", "markersize")]:
                if k in kwargs:
                    kwargs[k_new] = kwargs.pop(k)

            ax.step(new_xs, new_ys, where="post", **kwargs)

        return ax

    def fill(
        self,
        between: Pareto | None = None,
        *,
        ref: tuple[int, int] = (1, 1),
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        assert len(self.metrics) == 2

        if ax is None:
            ax = plt.gca()

        if between is None:
            _xs, _ys = zip(*sorted([e.point for e in self.evaluations]))
            xs = np.asarray([0, *_xs, 1])
            ys = np.asarray([1, *_ys, _ys[-1]])
            ax.fill_between(
                x=xs,
                y1=ys,
                y2=ref[1],
                step="post",
                **kwargs,
            )

        else:
            # ours = {e.point for e in self.evaluations}
            # theirs = {e.point for e in between.evaluations}

            raise NotImplementedError()
            # ax.fill_between(
            #    x=xs,
            #    y1=ys,
            #    y2=ref[1],
            #    step="post",
            #    **kwargs,
            # )

    def hypervolume(self, ref: tuple[int, int] = (1, 1)) -> float:
        """Assumes minimization and 2 bounded metrics in (0, 1)"""
        assert len(self.metrics) == 2

        df = self.df
        front = df.to_numpy()
        return hypervolume(front.copy(), ref)
