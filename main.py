from src.dataset import load_datasets
from src.evaluate import evaluate
from src.model import CNN
from src.train import train
import matplotlib.pyplot as plt
import torch, numpy as np, random, os, yaml
import copy

#---------------------Visualization Block------------------------------
def gatherAccuracies():
    """
    This function Collect final test accuracy for each experiment only first seed.
    Scans through all_plots/, where each experiment has its own subfolder,
    and reads accuracy.txt from each experiment folder.

    Returns:
        dict:
        {experiment_name (str) : accuracy (float)}
    """
    accuracies = {}
    for exp in os.listdir('all_plots'):
        accuracy_path = os.path.join('all_plots', exp, 'accuracy.txt')
        with open(accuracy_path, 'r') as f:
            acc = float(f.read().strip())
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

def run_experiment(config, seed):
    """
    Runs a single experiment for a given experiment onfiguration and seed.
    Notes:
    - Training and evaluation plots are saved only for the first seed.
    - Accuracy is saved per seed in: all_plots/<experiment>/accuracies/seed_<seed>.txt
    """
    setSeed(seed)

    if seed == seeds[0]: #save plots for the first seed only
        save_plots = True
    else:
        save_plots = False

    train_loader, test_loader = load_datasets(config, seed)
    model = CNN(config)
    print(f"Experiment: {config['experiment_type']} | Seed: {seed}") # Tracks ran experiment, for testing purposes
    train(model, train_loader, config, save_plots)
    accuracy = str(evaluate(model, test_loader, config, save_plots)) #this value is recorded in all_plots/<experiment>/accuracies/seed_<seed>.txt


    #make folder containg accuracies for each seed per experiment
    exp_directory = os.path.dirname(config['accuracy_path'])
    seed_dir = os.path.join(exp_directory, 'accuracies')
    os.makedirs(seed_dir, exist_ok = True)
    accuracy_path = os.path.join(seed_dir, f"seed_{seed}.txt")
    with open(accuracy_path, 'w') as f:
        f.write(str(accuracy))

#Main Execution Block
if __name__ == '__main__':
    """
    - Iterates over all experiment config files
    - For each experiment, the experiment is ran across all seeds
    - After every experiment completes:
        - Gathers accuracies
        - Generates summary plots
    """

    config_folder = "configs"
    config_files = [
        "baseline.yaml",
        "small_aug_reg.yaml",
        "small_no_aug.yaml",
        "small_with_aug.yaml",
        "small_rotation.yaml",
        "small_flip.yaml"
    ]


    for cfg in config_files: # run through all 6 different experiments
        config_path = os.path.join(config_folder, cfg)
        base_config = load_config(config_path)

        for seed in seeds:
            run_experiment(base_config, seed)

    # these gather and generate the accuracies and records it in visualizations/
    accuracies = gatherAccuracies()
    plotAccuracies(accuracies)
    seed_results = seed_statistics()
    plot_seed_results(seed_results)