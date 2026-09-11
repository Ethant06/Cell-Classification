"""
Forest plot: paired mean difference (limited aug − limited baseline) ± 95% CI.

Reads statistics/paired/summary.csv from compute_paired_statistics.py.
One row per experiment type; on each row, up to two CIs (cells + pneumonia).

Writes: statistics/paired/paired_difference_cis.png
       (with a transparent figure/axes background)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# Sans-serif stack (aligned with typical matplotlib / poster exports)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
            "sans-serif",
        ],
    }
)

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "statistics" / "paired" / "summary.csv"
OUT_PNG = ROOT / "statistics" / "paired" / "paired_difference_cis.png"

MEAN_MARKERSIZE = 13.5
CI_LINEWIDTH = 2.8
CI_CAPTHICK = 2.8
# X tick labels: base + 5px @ 96dpi + 6 pt
X_TICK_LABEL_PT = 10.0 + 5.0 * 72.0 / 96.0 + 6.0

CELL_COLOR = "#2E86AB"
LUNG_COLOR = "#E8871E"

# These suffixes must match PAIRS labels after their dataset prefix is removed.
# Each tuple is (summary.csv label suffix, y-axis display text).
ROW_ORDER: list[tuple[str, str]] = [
    ("flip vs limited baseline", "flip vs baseline"),
    ("rotation vs limited baseline", "rotated vs baseline"),
    ("aug vs limited baseline", "rotate/flip vs baseline"),
    ("erase vs limited baseline", "erase vs baseline"),
]

Y_DODGE = 0.08
# Vertical gap between experiment-type rows (smaller = tighter groups).
ROW_SPACING = 0.42


def load_rows() -> list[dict[str, str]]:
    """Load paired-comparison summaries or exit with a recovery instruction."""
    if not SUMMARY.is_file():
        raise SystemExit(f"Missing {SUMMARY}. Run: python scripts/compute_paired_statistics.py")
    with SUMMARY.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_suffix(row: dict[str, str]) -> str:
    """Remove the dataset-family prefix from a comparison label."""
    lab = row["label"]
    return lab.split(": ", 1)[-1] if ": " in lab else lab


def plot_ci(ax: Axes, row: dict[str, str], y: float, color: str) -> None:
    """Draw one mean difference with asymmetric 95% CI error bars.

    The row must contain ``mean_difference``, ``ci95_low``, and ``ci95_high``.
    """
    m = float(row["mean_difference"])
    lo = float(row["ci95_low"])
    hi = float(row["ci95_high"])
    xerr = np.vstack([m - lo, hi - m])
    ax.errorbar(
        m,
        y,
        xerr=xerr,
        fmt="o",
        capsize=7,
        elinewidth=CI_LINEWIDTH,
        capthick=CI_CAPTHICK,
        color=color,
        ecolor="black",
        markersize=MEAN_MARKERSIZE,
        markeredgewidth=1.05,
        markeredgecolor="black",
        zorder=3,
    )


def main() -> None:
    """Group available comparisons and write the confidence-interval PNG.

    Missing rows are skipped. Only the four augmentation types in ``ROW_ORDER``
    are shown; full-training baselines are intentionally excluded.
    """
    rows = load_rows()
    by_suffix: dict[str, dict[str, dict[str, str]]] = {"cells": {}, "pneumonia": {}}
    for r in rows:
        fam = r.get("dataset_family")
        if fam not in by_suffix:
            continue
        by_suffix[fam][label_suffix(r)] = r

    n = len(ROW_ORDER)
    row_y = np.arange(n) * ROW_SPACING
    fig, ax = plt.subplots(figsize=(10, 4.35), constrained_layout=True)
    ax.axvline(0.0, color="gray", linewidth=0.9, linestyle="--", zorder=1)

    for i, (csv_key, _) in enumerate(ROW_ORDER):
        b = row_y[i]
        if csv_key in by_suffix["cells"]:
            plot_ci(ax, by_suffix["cells"][csv_key], b - Y_DODGE, CELL_COLOR)
        if csv_key in by_suffix["pneumonia"]:
            plot_ci(ax, by_suffix["pneumonia"][csv_key], b + Y_DODGE, LUNG_COLOR)

    ax.set_yticks(row_y)
    ax.set_yticklabels([disp for _, disp in ROW_ORDER], fontsize=9)
    ax.invert_yaxis()
    margin = 0.11
    ax.set_ylim(row_y[0] - Y_DODGE - margin, row_y[-1] + Y_DODGE + margin)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", which="both", top=False, right=False)
    ax.tick_params(
        axis="x",
        which="major",
        top=False,
        labelsize=X_TICK_LABEL_PT,
        width=1.3,
        length=7,
    )

    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
