import os
import torch
from torch.utils.data import Subset, random_split, DataLoader
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch


def load_create_split(data_dir, test_ratio, seed = 42):
    """
    Creates or loads a fixed train/test split that is shared across
    all experiments and all random seeds. Each dataset gets its very own
    fixed train/test indices.

    - If saved split files exist, they are loaded from saved_splits/.
    - Otherwise, a new stratified split is created and saved as fixed.
    """
    dataset_name = os.path.basename(os.path.abspath(data_dir))
    split_dir = os.path.join("saved_splits", dataset_name)
    train_idx_path = os.path.join(split_dir, "train_indices.npy")
    test_idx_path = os.path.join(split_dir, "test_indices.npy")

    if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
        train_indices = np.load(train_idx_path)
        test_indices = np.load(test_idx_path)
        return train_indices, test_indices

    os.makedirs(split_dir, exist_ok=True)
    full_dataset = ImageFolder(root=data_dir)
    full_indices = np.arange(len(full_dataset))

    train_indices, test_indices = train_test_split(
        full_indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=full_dataset.targets
    )
    np.save(train_idx_path, train_indices)
    np.save(test_idx_path, test_indices)

    return train_indices, test_indices


def get_transform(config):
    """
    Returns image transformations for training and testing.
    Training Transform:
        - Depends on experiment type
        - May include augmentation depending on experiment configurations

    Test Transform:
        - Fixed across all experiments
        - No augmentation applied
    """

    exp = config['experiment_type']

    if exp == "small_aug":
        train_transform =  transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    elif exp == "small_aug_reg":
        train_transform =  transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p = 0.15),
            transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    elif exp == "small_rotation":
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    elif exp == "small_flip":
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    else:
        train_transform =  transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transform, test_transform


def load_datasets(config, seed):
    """
    -returns a test dataloader and a train dataloader
    -test dataloader is fixed for all experiments,
    -train dataloader will vary in data ratio depending on experiment type,
        and also data transformation

    Reduced subset sampling:
        - Varies across seeds
        - Is stratified to preserve class balance
        - Uses the SAME reduced subset across all "limited data" experiments
          for a given seed
    """

    train_transform, test_transform = get_transform(config)

    #train_indices will be used for subset sampling below
    train_indices, test_indices = load_create_split(
        data_dir = config['data_dir'],
        test_ratio = config['test_ratio']
    )

    full_train_dataset = ImageFolder(root=config['data_dir'], transform=train_transform)
    full_test_dataset = ImageFolder(root=config['data_dir'], transform=test_transform)

    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)


    """
    -reduced subset sampling for training data for experiments with limited data
    -subset changes per seed but stays the same for every limited data experiment
        -full_train_dataset.targets is a list of class labels for all images in the full dataset.
        -train_indices is the list of indices in training split
        -list comprehension gathers the labels corresponding to training samples.
    """
    if config['experiment_type'] != 'baseline':
        if config['data_ratio'] < 1.0:
            train_labels = np.array([full_train_dataset.targets[i] for i in train_indices])

            subset_size = int(len(train_dataset) * config['data_ratio'])

            # stratified subsampling from training split
            reduced_indices, _ = train_test_split(
                np.arange(len(train_dataset)),
                train_size=subset_size,
                stratify=train_labels,
                random_state=seed,
                shuffle=True
            )
            train_dataset = Subset(train_dataset, reduced_indices)

    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle= True) # this train_loader will only vary by seed
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False) #this test loader is fixed for all experiment
    return train_loader, test_loader