from src.dataset import load_datasets
from src.evaluate import evaluate
from src.model import CNN
from src.train import train
import matplotlib.pyplot as plt
import torch, numpy as np, random, os, yaml, argparse
import copy

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

#---------------------Visualization Block------------------------------
def gatherAccuracies():
    """
    Collect final test accuracy for each experiment (first seed only for bar chart).
    Reads accuracy.txt if present; otherwise uses mean of accuracies/seed_*.txt.
    """
    accuracies = {}
    for exp in os.listdir('all_plots'):
        exp_dir = os.path.join('all_plots', exp)
        accuracy_path = os.path.join(exp_dir, 'accuracy.txt')
        seed_dir = os.path.join(exp_dir, 'accuracies')
        if os.path.isfile(accuracy_path):
            with open(accuracy_path, 'r') as f:
                acc = float(f.read().strip())
        elif os.path.isdir(seed_dir):
            vals = []
            for fname in os.listdir(seed_dir):
                if fname.endswith('.txt'):
                    with open(os.path.join(seed_dir, fname), 'r') as f:
                        vals.append(float(f.read().strip()))
            acc = float(np.mean(vals)) if vals else 0.0
        else:
            continue
        accuracies[exp] = acc
    return accuracies


def plotAccuracies(accuracies):
    """
    accuracies parameter - gatherAccuracies result
    Plots a bar chart comparing test accuracy across experiments only first seed.
    """
    exp_names = list(accuracies.keys())
    exp_acc = list(accuracies.values())

    plt.figure(figsize = (11, 9))
    bars = plt.bar(exp_names, exp_acc, color = 'blue', edgecolor = 'black')
    plt.title('Test Accuracy Per Experiment', fontsize=14, weight='bold')
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracies')
    plt.xticks(rotation=20, ha='right', fontsize=8)
    for bar, val in zip(bars, exp_acc):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, val, ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join('visualizations', 'accuracy_summary_plot')
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, 'accuracy_comparison.png'))

def seed_statistics():
    """
    Computes mean and standard deviation of test accuracy
    across random seeds for each experiment.

    Result directory structure expected:
        all_plots/
            experiment_name/
                accuracies/
                    seed_12.txt
                    seed_5.txt

    Returns:
        dict:
        {experiment_name: (mean_accuracy, std_accuracy)}
    """
    results = {}
    base_dir = "all_plots"

    for exp in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, exp)
        seed_dir = os.path.join(exp_dir, 'accuracies')

        seed_values = []
        for seed_file in os.listdir(seed_dir):
            with open(os.path.join(seed_dir, seed_file), 'r') as f:
                seed_values.append(float(f.read().strip()))

        if seed_values:
            mean_acc = float(np.mean(seed_values))
            std_acc = float(np.std(seed_values))
            results[exp] = (mean_acc, std_acc)
    return results


# Cells (data2) vs lungs (data4) plot folders — order = x-axis groups, left/right bar per group.
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


def read_seed_mean_std(exp_folder: str):
    """
    Mean and std of test accuracy from all_plots/<exp_folder>/accuracies/seed_*.txt.
    Returns None if folder or seeds missing.
    """
    seed_dir = os.path.join("all_plots", exp_folder, "accuracies")
    if not os.path.isdir(seed_dir):
        return None
    seed_values = []
    for fname in os.listdir(seed_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(seed_dir, fname), "r") as f:
                seed_values.append(float(f.read().strip()))
    if not seed_values:
        return None
    return float(np.mean(seed_values)), float(np.std(seed_values))


def plot_seed_results(results=None):
    """
    Grouped bar chart: one x position per ablation setting; two bars (Cells vs Lungs)
    with mean ± std across seeds. Groups omitted if either modality has no seed files.
    ``results`` is kept for call-site compatibility but not used (stats read from disk).
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
            "skipping seed_ablation_comparison (.png / .svg)",
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

    def _label_bars(bar_container, means, stds):
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
    plt.savefig(base + ".svg", format="svg", facecolor="white")
    plt.close(fig)
#----------------------End of Visualization Block----------------------------



# ------------------Main Functions for running experiment-----------------------
seeds = [12, 5, 20, 44, 2, 7, 6, 33, 3, 10]

def setSeed(seed):
    """
    Set seed for
    - model initialization
    - data shuffling
    - reduced subset sampling
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(path):
    """
    Loads a YAML configuration file for an experiment.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config, seed, config_filename=None):
    """
    Runs a single experiment for a given experiment configuration and seed.
    Notes:
    - Training and evaluation plots are saved only for the first seed.
    - Accuracy is saved per seed in: all_plots/<experiment>/accuracies/seed_<seed>.txt
    - config_filename (e.g. pneumonia_flat_subset_aug.yaml) distinguishes runs that share the same experiment_type.
    Optional YAML key run_label: short display name; overrides filename in the log line.
    """
    setSeed(seed)

    if seed == seeds[0]: #save plots for the first seed only
        save_plots = True
    else:
        save_plots = False

    train_loader, test_loader = load_datasets(config, seed)
    model = CNN(config)
    run_id = config.get("run_label") or config_filename or "?"
    print(
        f"Run: {run_id} | experiment_type={config['experiment_type']} | Seed={seed}",
        flush=True,
    )
    train_accuracy = train(model, train_loader, config, save_plots)
    accuracy = str(evaluate(model, test_loader, config, save_plots)) #this value is recorded in all_plots/<experiment>/accuracies/seed_<seed>.txt


    # make folder containing test accuracies for each seed per experiment
    exp_directory = os.path.dirname(config['accuracy_path'])
    seed_dir = os.path.join(exp_directory, 'accuracies')
    os.makedirs(seed_dir, exist_ok=True)
    seed_path = os.path.join(seed_dir, f"seed_{seed}.txt")
    with open(seed_path, 'w') as f:
        f.write(str(accuracy))
    # keep accuracy.txt in sync with first seed so gatherAccuracies() works
    if seed == seeds[0]:
        with open(config['accuracy_path'], 'w') as f:
            f.write(str(accuracy))

    # make folder containg train accuracies for each seed per experiment
    train_seed_dir = os.path.join(exp_directory, 'train_accuracies')
    os.makedirs(train_seed_dir, exist_ok = True)
    train_accuracy_path = os.path.join(train_seed_dir, f"seed_{seed}.txt")
    with open(train_accuracy_path, 'w') as f:
        f.write(str(train_accuracy))

#Main Execution Block
if __name__ == '__main__':
    """
    - Iterates over all experiment config files
    - For each experiment, the experiment is ran across all seeds
    - After every experiment completes:
        - Gathers accuracies
        - Generates summary plots
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train/eval CNN experiments. Default run uses DEFAULT_CONFIG_FILES; "
            "each YAML sets its own data_dir (e.g. euploid/aneuploid ImageFolder root). "
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

        for seed in seeds:
            run_experiment(base_config, seed, cfg)

    # these gather and generate the accuracies and records it in visualizations/
    accuracies = gatherAccuracies()
    plotAccuracies(accuracies)
    seed_results = seed_statistics()
    plot_seed_results(seed_results)