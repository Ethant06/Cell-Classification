import os
import torch
from torch.utils.data import Subset, random_split, DataLoader
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch


def _split_matches_dataset(train_indices, test_indices, n_samples):
    """True if saved indices are a full partition of [0, n_samples) (same size as current ImageFolder)."""
    if n_samples <= 0:
        return False
    combined = np.concatenate([np.asarray(train_indices), np.asarray(test_indices)])
    if combined.size != n_samples:
        return False
    if combined.min() < 0 or combined.max() >= n_samples:
        return False
    return np.unique(combined).size == n_samples


def load_create_split(data_dir, test_ratio, seed = 42):
    """
    Creates or loads a fixed train/test split that is shared across
    all experiments and all random seeds. Each dataset gets its very own
    fixed train/test indices.

    - If saved split files exist, they are loaded from saved_splits/.
    - Otherwise, a new stratified split is created and saved as fixed.
    - If the image folder changed size (add/remove images), saved indices are invalid;
      a new split is created and overwrites the old files.
    """
    dataset_name = os.path.basename(os.path.abspath(data_dir))
    split_dir = os.path.join("saved_splits", dataset_name)
    train_idx_path = os.path.join(split_dir, "train_indices.npy")
    test_idx_path = os.path.join(split_dir, "test_indices.npy")

    full_dataset = ImageFolder(root=data_dir)
    n = len(full_dataset)

    if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
        train_indices = np.load(train_idx_path)
        test_indices = np.load(test_idx_path)
        if _split_matches_dataset(train_indices, test_indices, n):
            return train_indices, test_indices

    os.makedirs(split_dir, exist_ok=True)
    full_indices = np.arange(n)

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

    elif exp == "small_erase":
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15),
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


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def load_datasets(config, seed):
    """
    Returns (train_loader, test_loader).
    - Test dataloader is fixed for all experiments.
    - Train dataloader varies by data ratio and transforms.

    When config has data_dir_train and data_dir_test (e.g. pneumonia with official split),
    uses those dirs directly to avoid patient/sample leakage from merging then re-splitting.
    Otherwise uses data_dir and load_create_split for a fixed 70/30 stratified split.

    Reduced subset sampling (limited-data experiments):
        - Stratified subsample of the training set; same subset across all limited-data
          experiments for a given seed.
    """
    train_transform, test_transform = get_transform(config)

    #train_indices will be used for subset sampling below
    train_indices, test_indices = load_create_split(
        data_dir = config['data_dir'],
        test_ratio = config['test_ratio']
    )

    # No transforms applied to the full dataset
    full_dataset = ImageFolder(root=config['data_dir'])

    # Create subsets first
    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)

    # Apply transforms to the subsets
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    test_dataset = TransformedSubset(test_subset, transform=test_transform)


    """
    -reduced subset sampling for training data for experiments with limited data
    -subset changes per seed but stays the same for every limited data experiment
        -full_train_dataset.targets is a list of class labels for all images in the full dataset.
        -train_indices is the list of indices in training split
        -list comprehension gathers the labels corresponding to training samples.
    """
    if config['experiment_type'] != 'baseline':
        if config['data_ratio'] < 1.0:
            # Get labels from the original train_subset
            train_labels = np.array([train_subset.dataset.targets[i] for i in train_subset.indices])

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
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False) #this test loader is fixed for all experiment
    return train_loader, test_loader