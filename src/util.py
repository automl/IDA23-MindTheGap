from __future__ import annotations

import operator

import numpy as np


def pareto_front(
    costs: np.ndarray,
    optimistic: bool = True,
) -> list[bool]:
    """Pareto front of a set of costs (lower is better).

    Parameters
    ----------
    costs: np.ndarray (n_points, n_costs)
        The associated score of each point considered

    optimistic: bool = True
        Whether to generate the optimistic pareto front possible
        or the pessimistic one. Ideally, the costs are already
        near the optimal and so considering the worst will not
        cause a dramatic change

    Returns
    -------
    list[bool]
        A list of bools indicating whether each point is on the pareto
        front or not
    """
    is_pareto = np.ones(costs.shape[0], dtype=bool)

    # We are making the assumption each cost is better when minimized
    # this assumption is stated in `def validate_for_moo_benchmark` and is
    # called in `def evaluate`.
    op = operator.lt if optimistic else operator.gt

    for i, c in enumerate(costs):
        if is_pareto[i]:
            points_mask = np.any(op(costs, c), axis=1)
            is_pareto = np.logical_and(points_mask, is_pareto)
            is_pareto[i] = True  # keep self

    return list(is_pareto)
