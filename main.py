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
]

# Pneumonia flat data4/ (class folders); each YAML writes under all_plots/pneumonia_* per accuracy_path.
PNEUMONIA_CONFIG_FILES = [
    "pneumonia_flat_baseline.yaml",
    "pneumonia_subset_baseline.yaml",
    "pneumonia_flat_subset_aug.yaml",
    "pneumonia_flat_subset_aug_reg.yaml",
    "pneumonia_flat_subset_flip.yaml",
    "pneumonia_flat_subset_rotation.yaml",
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


def plot_seed_results(results):
    """
    Plots mean test accuracy with error bars (± std) across seeds.
    """
    exp_names = list(results.keys())
    means = [results[e][0] for e in exp_names]
    stds  = [results[e][1] for e in exp_names]

    plt.figure(figsize=(12, 9))
    bars = plt.bar(exp_names, means, yerr = stds, capsize = 6, color='royalblue', edgecolor='black')

    plt.title("Ablation Study: Mean Accuracy ± Std Across Seeds", fontsize=15, weight='bold')
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.5)
    plt.xticks(rotation=35, ha='right', fontsize=8)

    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, mean + 0.02, f"{mean:.3f}", ha = 'center', va = 'bottom')

    save_dir = os.path.join("visualizations", "ablation_study")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "seed_ablation_comparison.png"))
#----------------------End of Visualization Block----------------------------



# ------------------Main Functions for running experiment-----------------------
seeds = [12, 5, 20, 44]

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
        help="Run only pneumonia YAMLs (data4/); outputs under all_plots/pneumonia_*.",
    )
    args = parser.parse_args()

    config_folder = "configs"
    config_files = PNEUMONIA_CONFIG_FILES if args.pneumonia else DEFAULT_CONFIG_FILES

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