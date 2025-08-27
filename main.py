from src.dataset import load_datasets
from src.evaluate import evaluate
from src.model import CNN
from src.train import train
import yaml
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(config):

    train_loader, test_loader = load_datasets(config)
    
    model = CNN(config)
    
    train(model, train_loader, config)
    
    evaluate(model, test_loader, config)
    
    
    
if __name__ == '__main__':
    
    
    config_folder = "configs"
    config_files = [
        "baseline.yaml",
        "small_aug_reg.yaml",
        "small_no_aug.yaml",
        "small_with_aug.yaml"
    ]
    
    # run through all 4 different experiments
    for cfg in config_files:
        config_path = os.path.join(config_folder, cfg)
        config = load_config(config_path)
        run_experiment(config)
    