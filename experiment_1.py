from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import chain

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.metrics import precision_error, recall_error
from src.randomized_trainer import RandomizedTrainer
from src.random_forest import RandomForest

sns.set_context("paper", font_scale=0.6)

# Presets
chosen = "final"
presets = {
    "one": (1337, 5, (0.6, 0.2, 0.2)),
    "two": (1337, 10, (0.6, 0.2, 0.2)),
    "three": (1337, 20, (0.6, 0.2, 0.2)),
    "four": (1991, 5, (0.6, 0.2, 0.2)),
    "five": (1991, 10, (0.6, 0.2, 0.2)),
    "six": (1337, 5, (0.8, 0.1, 0.1)),
    "seven": (1337, 10, (0.8, 0.1, 0.1)),
    "eight": (1337, 20, (0.8, 0.1, 0.1)),
    "nine": (10, 5, (0.8, 0.1, 0.1)),
    #
    "final": (1337, 40, (0.6, 0.2, 0.2)),
}
seed, n_estimators, splits = presets[chosen]

argparser = ArgumentParser()
argparser.add_argument(
    "--task",
    type=int,
    default=31,
    help="OpenML task ID, default credit-g (31)"
)
argparser.add_argument(
    "--n-estimators",
    type=int,
    default=n_estimators,
    help="Number of random configurations to train"
)
args = argparser.parse_args()
n_estimators = args.n_estimators

# Plotting size stuffs
figsize = (11.6, 4)
dpi = 300
plot_backend = None  # "QTAgg"
plot_offset = 0.05
title_size = 18
label_size = 16
tick_size = 12

task_id = args.task
metrics = [precision_error, recall_error]
metric_names = ".".join([m.name for m in metrics])

setup_name = f"{task_id}_{seed}_{n_estimators}_{splits}_{metric_names}"


@dataclass
class Split:
    features: pd.DataFrame
    labels: pd.Series


# Get the xs, ys
task = openml.tasks.get_task(task_id)
next_xs, next_ys = task.get_X_and_y(dataset_format="dataframe")
# Replace strings etc in the targets by 0 and 1 (as expected by sklearn's precision)
uniques = next_ys.unique()
counts = sorted([(np.sum(u == next_ys), u) for u in uniques])
next_ys = pd.Series([1 if ny == counts[0][1] else 0 for ny in next_ys], dtype="category")

# Size in total and sample sizes per splits
N = len(next_xs)
sample_sizes = tuple(int(s * len(next_xs)) for s in splits)

# datasplits
datasplits = []
for s in sample_sizes[:-1]:
    (xs, next_xs, ys, next_ys) = train_test_split(
        *(next_xs, next_ys),
        train_size=s,
        stratify=next_ys,
        random_state=seed,
    )
    datasplits.append(Split(xs, ys))

datasplits.append(Split(next_xs, next_ys))

# Take out the datasplits
train, val, test = datasplits


# Create the random search trainer
model_class = RandomForest
rs = RandomizedTrainer(
    model_class=model_class,
    id=f"RS_{model_class}_{setup_name}",
    n=n_estimators,
    seed=seed,
)

if rs.modelpath.exists():
    rs = rs.load()
else:
    rs.fit(train.features, train.labels)
    rs.save()

# Finally, evaluate
val_scores = rs.evaluate(val.features, val.labels, metrics=metrics, name="validation")
test_scores = rs.evaluate(test.features, test.labels, metrics=metrics, name="test")

# These are essentially some runs that receive the worst scores runs
# and we remove them for the sake of the plot, in practice these do not make a
# difference ot any results shown in the plots or with hypervolume calculation
deletables = {
    id for id, e in chain(val_scores.items(), test_scores.items()) if e.point == (1.0, 1.0)
}
for id in deletables:
    del val_scores[id]
    del test_scores[id]

val_pareto = val_scores.pareto_front()
test_pareto = test_scores.pareto_front()

val_pareto_on_test_scores = test_scores.only(val_pareto)

val_pareto_on_test_optimistic = val_pareto_on_test_scores.pareto_front(optimistic=True)
val_pareto_on_test_pessimistic = val_pareto_on_test_scores.pareto_front(optimistic=False)

# Figures
if plot_backend:
    matplotlib.use(plot_backend)

fig, (left_ax, middle_ax, right_ax) = plt.subplots(
    nrows=1,
    ncols=3,
    sharey=True,
    figsize=figsize,
)
fig.supxlabel(metrics[0].name.replace("_", " ").capitalize(), fontsize=label_size)
fig.supylabel(metrics[1].name.replace("_", " ").capitalize(), fontsize=label_size)

left_ax.set_title("Validation", fontsize=title_size)
middle_ax.set_title("Validation → Test", fontsize=title_size)
right_ax.set_title("Test", fontsize=title_size)
for ax in (left_ax, middle_ax, right_ax):
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    ax.set_box_aspect(1)

def rgb(r: int, g: int, b: int) -> str:
    return "#%02x%02x%02x" % (r, g, b)


c_chocolate = rgb(217, 95, 2)
c_darkcyan = rgb(27, 158, 119)
c_darkgray = rgb(176, 176, 176)
c_deeppink = rgb(231, 41, 138)
c_light_gray = rgb(204, 204, 204)
c_light_slate_gray = rgb(117, 112, 179)
c_green = rgb(102, 166, 30)
c_purple = rgb(117, 112, 179)

alpha = 0.25
arrow_alpha = 1.0

styles = {
    "val_points": dict(s=15, marker="o", edgecolors=c_chocolate, facecolors="none"),
    "val_pareto": dict(s=4, marker="o", color=c_chocolate, linestyle="dotted", linewidth=2),
    "test_points": dict(s=15, marker="o", edgecolors="black", facecolors="none"),
    "test_pareto": dict(s=4, marker="o", color="black", linestyle="-", linewidth=2),
    "moved_points": dict(s=15, marker="o", color=c_darkcyan),
    "optimistic_pareto": dict(s=4, marker="o", color=c_darkcyan, linewidth=2),
    "pessimistic_pareto": dict(s=4, marker="o", color=c_deeppink, linewidth=2),
    "arrows": dict(color="black", width=0.005, zorder=10),
}

# For the validation set
# ======================
ax = left_ax

# Highlight val stuff
val_scores.plot(ax=ax, **styles["val_points"])
val_pareto.plot(ax=ax, **styles["val_pareto"])

# Faded test stuff
# test_scores.plot(ax=ax, alpha=alpha, **styles["test_points"])
test_pareto.plot(ax=ax, alpha=alpha, **styles["test_pareto"])

# For the valid and test set together
# ===================================
ax = middle_ax

# Highlight val pareto front and how things moved
val_pareto_on_test_scores.plot(ax=ax, **styles["moved_points"])
val_pareto.plot(ax=ax, **styles["val_pareto"])
val_pareto.plot_arrows(to=test_scores, ax=ax, alpha=arrow_alpha, **styles["arrows"])

# Show the test pareto but faded
test_pareto.plot(ax=ax, alpha=alpha, **styles["test_pareto"])
# test_scores.plot( ax=ax, alpha=alpha, **styles["test_points"])

# The optimistic and pesimistic pareto fronts
# ===========================================
ax = right_ax

# Show the optimistic and pessimistic pareto fronts
val_pareto_on_test_optimistic.plot(ax=ax, **styles["optimistic_pareto"])
val_pareto_on_test_optimistic.fill(ax=ax, alpha=alpha, color=styles["optimistic_pareto"]["color"])

val_pareto_on_test_pessimistic.plot(ax=ax, **styles["pessimistic_pareto"])
val_pareto_on_test_pessimistic.fill(ax=ax, alpha=alpha, color=styles["pessimistic_pareto"]["color"])

# Show the original validation and test pareto faded for reference
val_pareto.plot(ax=ax, alpha=alpha, **styles["val_pareto"])
test_pareto.plot(ax=ax, alpha=alpha, **styles["test_pareto"])

# Show how the val pareto moved
val_pareto_on_test_optimistic.plot_arrows(frm=val_pareto, ax=ax, alpha=arrow_alpha, **styles["arrows"])
val_pareto_on_test_pessimistic.plot_arrows(frm=val_pareto, ax=ax, alpha=arrow_alpha, **styles["arrows"])

# Final touches
# =============
points = [
    e.point
    for e in chain(
        val_scores.evaluations,
        test_pareto.evaluations,
        val_pareto_on_test_optimistic.evaluations,
        val_pareto_on_test_pessimistic.evaluations,
    )
]
xs, ys = zip(*points)
min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)

dx = abs(max_x - min_x)
dy = abs(max_y - min_y)


for ax in (left_ax, middle_ax, right_ax):
    ax.set_xlim(min_x - dx * plot_offset, max_x + dx * plot_offset)
    ax.set_ylim(min_y - dy * plot_offset, max_y + dy * plot_offset)
    # We're adding the legend in tex code
    # ax.legend()

fig.tight_layout(rect=[0, 0.02, 1, 0.98])

plt.savefig(f"./figures/experiment_1_{chosen}.png", bbox_inches="tight", pad_inches=0, dpi=dpi)
plt.savefig(f"./figures/experiment_1_{chosen}.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)

