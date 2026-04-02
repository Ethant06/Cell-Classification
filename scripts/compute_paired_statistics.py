r"""
================================================================================
WHAT THIS SCRIPT DOES (walkthrough)
================================================================================

1) all_plots/ is NOT modified
   - Your training pipeline still writes accuracies there.
   - main.py bar charts still use all_plots. We only READ numbers from it.

2) Where numbers come from
   - Path: all_plots/<experiment_folder>/accuracies/seed_<N>.txt
   - Each file holds one float: test accuracy for that experiment at that seed.

3) What a "pair" is
   - One folder = no-aug subset baseline, another folder = aug (e.g. flip).
   - For each seed that exists in BOTH folders, we form ONE paired difference:
        difference = acc_aug - acc_baseline
   - Same seed → same train/test subset draw in your pipeline, so pairing is fair.

4) What we compute per pair
   - n = number of common seeds
   - mean_diff = average of those differences
   - 95% confidence interval for that mean (paired t, df = n-1)

5) What we write (under statistics/paired/)
   - summary.csv           — machine-readable, one row per pair
   - summary_table.txt    — SAME info, aligned text table (easy to open in editor)
   - long_format.csv      — one row per (pair, seed) for spreadsheets
   - by_comparison/<id>/differences.csv — per-pair seed table

6) After running this, make the figure:
   python scripts/plot_paired_cis.py

Usage:
  python scripts/compute_paired_statistics.py
================================================================================
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ALL_PLOTS = ROOT / "all_plots"
OUT = ROOT / "statistics" / "paired"

# Each tuple: (baseline folder under all_plots, aug folder, id for files, human label)
PAIRS: list[tuple[str, str, str, str]] = [
    ("subset_baseline_plot", "subset_aug_plot", "cells_subset_aug_vs_baseline", "cells: aug vs baseline"),
    ("subset_baseline_plot", "subset_flip_plot", "cells_subset_flip_vs_baseline", "cells: flip vs baseline"),
    ("subset_baseline_plot", "subset_rotation_plot", "cells_subset_rotation_vs_baseline", "cells: rotation vs baseline"),
    ("subset_baseline_plot", "subset_aug_reg_plot", "cells_subset_aug_reg_vs_baseline", "cells: aug+reg vs baseline"),
    ("pneumonia_subset_baseline", "pneumonia_flat_subset_aug_plot", "pneumonia_subset_aug_vs_baseline", "pneumonia: aug vs baseline"),
    ("pneumonia_subset_baseline", "pneumonia_flat_subset_flip_plot", "pneumonia_subset_flip_vs_baseline", "pneumonia: flip vs baseline"),
    ("pneumonia_subset_baseline", "pneumonia_flat_subset_rotation_plot", "pneumonia_subset_rotation_vs_baseline", "pneumonia: rotation vs baseline"),
]


def family(comparison_id: str) -> str:
    return "pneumonia" if comparison_id.startswith("pneumonia") else "cells"


def read_one_experiment_accuracies(exp_name: str) -> dict[int, float]:
    """Map seed -> test accuracy for one experiment folder."""
    folder = ALL_PLOTS / exp_name / "accuracies"
    if not folder.is_dir():
        return {}
    out: dict[int, float] = {}
    for path in folder.glob("seed_*.txt"):
        m = re.match(r"seed_(\d+)\.txt$", path.name)
        if not m:
            continue
        try:
            out[int(m.group(1))] = float(path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return out


def load_everything() -> dict[str, dict[int, float]]:
    """exp_folder_name -> {seed: accuracy}"""
    if not ALL_PLOTS.is_dir():
        return {}
    all_exp: dict[str, dict[int, float]] = {}
    for d in ALL_PLOTS.iterdir():
        if d.is_dir():
            all_exp[d.name] = read_one_experiment_accuracies(d.name)
    return all_exp


def t_ci_on_mean(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Returns mean, std, ci_low, ci_high for the mean of `values` (paired differences)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    from scipy import stats

    t_star = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    half = float(t_star * sem)
    return mean, std, mean - half, mean + half


def build_summary_and_long(
    exps: dict[str, dict[int, float]],
) -> tuple[list[dict], list[dict]]:
    long_rows: list[dict] = []
    summary_rows: list[dict] = []

    for baseline_name, aug_name, comp_id, label in PAIRS:
        b = exps.get(baseline_name, {})
        a = exps.get(aug_name, {})
        seeds = sorted(set(b) & set(a))
        if len(seeds) < 2:
            print(f"[skip] {comp_id}: need >=2 seeds in both folders; got {seeds}")
            continue

        base_acc = np.array([b[s] for s in seeds])
        aug_acc = np.array([a[s] for s in seeds])
        diff = aug_acc - base_acc
        mn, sd, lo, hi = t_ci_on_mean(diff)

        summary_rows.append(
            {
                "dataset_family": family(comp_id),
                "comparison_id": comp_id,
                "label": label,
                "baseline_folder": baseline_name,
                "aug_folder": aug_name,
                "n_seeds": len(seeds),
                "seeds_used": ",".join(map(str, seeds)),
                "mean_difference": mn,
                "std_difference": sd,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_method": "paired_t_mean_diff",
            }
        )

        for s, bv, av, dv in zip(seeds, base_acc.tolist(), aug_acc.tolist(), diff.tolist()):
            long_rows.append(
                {
                    "dataset_family": family(comp_id),
                    "comparison_id": comp_id,
                    "label": label,
                    "baseline_folder": baseline_name,
                    "aug_folder": aug_name,
                    "seed": s,
                    "baseline_value": bv,
                    "aug_value": av,
                    "difference_aug_minus_baseline": dv,
                }
            )

        # per-pair detail folder
        sub = OUT / "by_comparison" / comp_id
        sub.mkdir(parents=True, exist_ok=True)
        with (sub / "differences.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["seed", "baseline_value", "aug_value", "difference_aug_minus_baseline"],
            )
            w.writeheader()
            for s, bv, av, dv in zip(seeds, base_acc.tolist(), aug_acc.tolist(), diff.tolist()):
                w.writerow(
                    {"seed": s, "baseline_value": bv, "aug_value": av, "difference_aug_minus_baseline": dv}
                )

    return long_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_table_txt(path: Path, summary_rows: list[dict]) -> None:
    """Human-readable aligned table: open in any editor or paste into notes."""
    if not summary_rows:
        return
    lines = [
        "Paired comparisons: difference = aug test_accuracy - baseline test_accuracy",
        "95% CI = paired t interval on the MEAN of those differences (df = n_seeds - 1).",
        "CI includes 0  ~  no strong evidence of systematic gain or loss.",
        "",
    ]
    cols = ["family", "label", "n", "mean_diff", "ci95_low", "ci95_high", "seeds"]
    lines.append("\t".join(cols))
    lines.append("-" * 110)
    for r in summary_rows:
        lines.append(
            "\t".join(
                [
                    str(r["dataset_family"]),
                    str(r["label"])[:36],
                    str(r["n_seeds"]),
                    f'{float(r["mean_difference"]):.4f}',
                    f'{float(r["ci95_low"]):.4f}',
                    f'{float(r["ci95_high"]):.4f}',
                    str(r["seeds_used"]),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    exps = load_everything()
    if not exps:
        print("No data under all_plots/. Run training first.")
        return

    long_rows, summary_rows = build_summary_and_long(exps)
    OUT.mkdir(parents=True, exist_ok=True)

    write_csv(OUT / "long_format.csv", long_rows)
    write_csv(OUT / "summary.csv", summary_rows)
    write_summary_table_txt(OUT / "summary_table.txt", summary_rows)

    print(f"Wrote summaries under {OUT}")
    print("  Open statistics/paired/summary_table.txt for the quick table.")
    print("  Then: python scripts/plot_paired_cis.py")


if __name__ == "__main__":
    main()
