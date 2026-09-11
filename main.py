"""Run CNN experiments and generate the cross-dataset ablation summary.

Each selected YAML configuration is trained with every seed in ``SEEDS``. Per-seed
test and training accuracies are written under ``all_plots/``; the collected test
accuracies then produce the retained ablation-study PNG.
"""

import argparse
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.dataset import load_datasets
from src.evaluate import evaluate
from src.model import CNN
from src.train import train

DEFAULT_CONFIG_FILES = [
    "baseline.yaml",
    "small_aug_reg.yaml",
    "small_no_aug.yaml",
    "small_with_aug.yaml",
    "small_rotation.yaml",
    "small_flip.yaml",
    "small_erase.yaml",
]

# Pneumonia flat data4/ (class folders); each YAML writes under all_plots/pneumonia_* per accuracy_path.
PNEUMONIA_CONFIG_FILES = [
    "pneumonia_flat_baseline.yaml",
    "pneumonia_subset_baseline.yaml",
    "pneumonia_flat_subset_aug.yaml",
    "pneumonia_flat_subset_aug_reg.yaml",
    "pneumonia_flat_subset_flip.yaml",
    "pneumonia_flat_subset_rotation.yaml",
    "pneumonia_flat_subset_erase.yaml",
]

# Each tuple is: display label, cell result folder, pneumonia result folder.
# "Original" is the limited-data, no-augmentation control—not the full baseline.
ABLATION_GROUP_BARS = [
    ("Original", "subset_baseline_plot", "pneumonia_subset_baseline"),
    ("Flip", "subset_flip_plot", "pneumonia_flat_subset_flip_plot"),
    ("Rotated", "subset_rotation_plot", "pneumonia_flat_subset_rotation_plot"),
    ("Rotate/Flip", "subset_aug_plot", "pneumonia_flat_subset_aug_plot"),
    ("Erase", "subset_erase_plot", "pneumonia_flat_subset_erase_plot"),
]
CELL_BAR_COLOR = "#2E86AB"
LUNG_BAR_COLOR = "#E8871E"
# Typography (~24px experiment ticks; ~21px bar-top decimal accuracy)
POSTER_24PX_PT = 24.0 * 72.0 / 96.0
POSTER_21PX_PT = 21.0 * 72.0 / 96.0
_PX96 = 72.0 / 96.0  # CSS px @ 96dpi → matplotlib points
BAR_VALUE_LABEL_PT = POSTER_21PX_PT + 2.0 * _PX96
EXPERIMENT_TICK_FONT_PT = POSTER_24PX_PT
# X-axis: +2 px then +3 px vs original 24px-equivalent tick size
EXPERIMENT_TICK_FONT_PT_XAXIS = EXPERIMENT_TICK_FONT_PT + 5.0 * _PX96


def read_seed_mean_std(exp_folder: str) -> tuple[float, float] | None:
    """Return population mean and standard deviation for an experiment's seed accuracies.

    Only files named ``seed_*.txt`` are included. NumPy's default ``ddof=0`` is
    intentional because the chart summarizes this fixed set of experiment seeds.

    Args:
        exp_folder: Experiment directory name beneath ``all_plots``.

    Returns:
        ``(mean, standard_deviation)`` when seed files exist, otherwise ``None``.
    """
    seed_dir = os.path.join("all_plots", exp_folder, "accuracies")
    if not os.path.isdir(seed_dir):
        return None
    seed_values = []
    for fname in os.listdir(seed_dir):
        if fname.startswith("seed_") and fname.endswith(".txt"):
            with open(os.path.join(seed_dir, fname), "r") as f:
                seed_values.append(float(f.read().strip()))
    if not seed_values:
        return None
    return float(np.mean(seed_values)), float(np.std(seed_values))


def plot_seed_results() -> None:
    """Write the grouped cell-versus-pneumonia ablation chart.

    Each bar shows mean test accuracy across seeds and each error bar shows one
    population standard deviation. A group is omitted unless both datasets have
    seed results, preventing incomplete side-by-side comparisons. The PNG is
    written to ``visualizations/ablation_study/seed_ablation_comparison.png``.
    """
    group_labels = []
    cells_means, cells_stds = [], []
    lungs_means, lungs_stds = [], []

    for title, cells_exp, lungs_exp in ABLATION_GROUP_BARS:
        cs = read_seed_mean_std(cells_exp)
        ls = read_seed_mean_std(lungs_exp)
        if cs is None or ls is None:
            continue
        group_labels.append(title)
        cells_means.append(cs[0])
        cells_stds.append(cs[1])
        lungs_means.append(ls[0])
        lungs_stds.append(ls[1])

    if not group_labels:
        print(
            "plot_seed_results: no paired ablation folders with seed accuracies; "
            "skipping seed_ablation_comparison.png",
            flush=True,
        )
        return

    n = len(group_labels)
    x = np.arange(n)
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 7))
    bars_cells = ax.bar(
        x - width / 2,
        cells_means,
        width,
        yerr=cells_stds,
        capsize=5,
        color=CELL_BAR_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )
    bars_lungs = ax.bar(
        x + width / 2,
        lungs_means,
        width,
        yerr=lungs_stds,
        capsize=5,
        color=LUNG_BAR_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )

    def _label_bars(
        bar_container: Any, means: Sequence[float], stds: Sequence[float]
    ) -> None:
        """Place a three-decimal mean label above each error bar."""
        for rect, m, s in zip(bar_container.patches, means, stds):
            y = float(m) + float(s) + 0.018
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y,
                f"{float(m):.3f}",
                ha="center",
                va="bottom",
                fontsize=BAR_VALUE_LABEL_PT,
            )

    _label_bars(bars_cells, cells_means, cells_stds)
    _label_bars(bars_lungs, lungs_means, lungs_stds)

    ymax = max(
        m + s for m, s in zip(cells_means + lungs_means, cells_stds + lungs_stds)
    )
    ax.set_ylim(0.0, max(1.22, ymax + 0.14))
    ax.set_xticks(x)
    ax.set_xticklabels(
        group_labels,
        rotation=18,
        ha="right",
        fontsize=EXPERIMENT_TICK_FONT_PT_XAXIS,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.02)
    fig.tight_layout()

    save_dir = os.path.join("visualizations", "ablation_study")
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, "seed_ablation_comparison")
    plt.savefig(base + ".png", dpi=150, facecolor="white")
    plt.close(fig)
# Completed-study seeds. The first seed also controls single-copy text outputs.
SEEDS = [12, 5, 20, 44, 2, 7, 6, 33, 3, 10]


def set_seed(seed: int) -> None:
    """Seed PyTorch, NumPy, and Python random-number generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and return one YAML experiment configuration.

    Raises:
        ValueError: If the YAML document is empty or is not a mapping.
    """
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def run_experiment(
    config: Mapping[str, Any], seed: int, config_filename: str | None = None
) -> None:
    """Train and evaluate one configuration with one random seed.

    The function records test and training accuracy in the experiment directory
    derived from ``accuracy_path``. The first seed additionally refreshes
    ``accuracy.txt`` and the human-readable classification report.

    Args:
        config: Experiment settings loaded from YAML. Required keys are
            ``data_dir``, ``test_ratio``, ``experiment_type``, ``data_ratio``,
            ``batch_size``, ``epochs``, ``lr``, ``momentum``,
            ``accuracy_path``, and ``report_path``.
        seed: Seed controlling model initialization, shuffling, and subsampling.
        config_filename: Optional filename used to identify the run in logs.

    Notes:
        Stored test accuracy is ordinary classification accuracy. Balanced
        accuracy and macro-F1 are printed and included in first-seed reports.
    """
    set_seed(seed)

    save_report = seed == SEEDS[0]

    train_loader, test_loader = load_datasets(config, seed)
    model = CNN(config)
    run_id = config.get("run_label") or config_filename or "?"
    print(
        f"Run: {run_id} | experiment_type={config['experiment_type']} | Seed={seed}",
        flush=True,
    )
    train_accuracy = train(model, train_loader, config)
    accuracy = evaluate(model, test_loader, config, save_report)


    # Keep scalar results as plain text so downstream statistics remain transparent.
    exp_directory = os.path.dirname(config['accuracy_path'])
    seed_dir = os.path.join(exp_directory, 'accuracies')
    os.makedirs(seed_dir, exist_ok=True)
    seed_path = os.path.join(seed_dir, f"seed_{seed}.txt")
    with open(seed_path, 'w') as f:
        f.write(str(accuracy))
    if seed == SEEDS[0]:
        with open(config['accuracy_path'], 'w') as f:
            f.write(str(accuracy))

    train_seed_dir = os.path.join(exp_directory, 'train_accuracies')
    os.makedirs(train_seed_dir, exist_ok = True)
    train_accuracy_path = os.path.join(train_seed_dir, f"seed_{seed}.txt")
    with open(train_accuracy_path, 'w') as f:
        f.write(str(train_accuracy))

def main() -> None:
    """Parse command-line options, run requested experiments, and plot the summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Train/eval CNN experiments. Default run uses DEFAULT_CONFIG_FILES; "
            "each YAML selects data2/ (cell classes) or data4/ (lung classes). "
            "Use --pneumonia for pneumonia YAMLs only."
        )
    )
    parser.add_argument(
        "-p",
        "--pneumonia",
        action="store_true",
        help=(
            "Run only pneumonia YAMLs (data4/). "
            "Default runs cells then pneumonia configs so the paired ablation bar chart has both bars."
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "Optional list of config filenames (under configs/) to run exclusively. "
            "Example: --configs small_erase.yaml pneumonia_flat_subset_erase.yaml"
        ),
    )
    args = parser.parse_args()

    config_folder = "configs"
    if args.configs:
        config_files = args.configs
    else:
        config_files = (
            PNEUMONIA_CONFIG_FILES if args.pneumonia else DEFAULT_CONFIG_FILES + PNEUMONIA_CONFIG_FILES
        )

    for cfg in config_files:
        config_path = os.path.join(config_folder, cfg)
        base_config = load_config(config_path)

        for seed in SEEDS:
            run_experiment(base_config, seed, cfg)

    plot_seed_results()


if __name__ == '__main__':
    main()