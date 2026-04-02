"""
Plot mean paired difference (aug - baseline) with 95% CI from statistics/paired/summary.csv

Reads: statistics/paired/summary.csv  (run compute_paired_statistics.py first)
Writes: statistics/paired/paired_difference_cis.png

Top panel = cells, bottom panel = pneumonia. Vertical line at 0.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "statistics" / "paired" / "summary.csv"
OUT_PNG = ROOT / "statistics" / "paired" / "paired_difference_cis.png"


def load_rows() -> list[dict[str, str]]:
    if not SUMMARY.is_file():
        raise SystemExit(f"Missing {SUMMARY}. Run: python scripts/compute_paired_statistics.py")
    with SUMMARY.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def panel(ax, rows: list[dict], title: str) -> None:
    if not rows:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    labels = [r["label"].split(": ", 1)[-1] if ": " in r["label"] else r["label"] for r in rows]
    means = np.array([float(r["mean_difference"]) for r in rows])
    lo = np.array([float(r["ci95_low"]) for r in rows])
    hi = np.array([float(r["ci95_high"]) for r in rows])
    y = np.arange(len(rows))
    xerr = np.vstack([means - lo, hi - means])

    ax.axvline(0.0, color="gray", linewidth=0.9, linestyle="--")
    ax.errorbar(means, y, xerr=xerr, fmt="o", capsize=4, color="steelblue", ecolor="black", markersize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean paired difference (aug − baseline) on test accuracy ± 95% CI")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)


def main() -> None:
    rows = load_rows()
    cells = [r for r in rows if r.get("dataset_family") == "cells"]
    pneu = [r for r in rows if r.get("dataset_family") == "pneumonia"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    panel(ax0, cells, "Euploid / aneuploid (subset baseline vs augmentation)")
    panel(ax1, pneu, "Pneumonia (subset baseline vs augmentation)")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
