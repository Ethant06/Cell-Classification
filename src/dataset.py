"""Dataset splitting, augmentation, and DataLoader construction.

Both image collections use torchvision's ``ImageFolder`` layout. A fixed,
stratified train/test partition is persisted under ``saved_splits/`` for each
dataset, while limited-data experiments draw a seed-specific stratified subset
from the shared training partition.
"""

import os
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _split_matches_dataset(
    train_indices: np.ndarray, test_indices: np.ndarray, n_samples: int
) -> bool:
    """Return whether saved indices exactly partition the current dataset."""
    if n_samples <= 0:
        return False
    combined = np.concatenate([np.asarray(train_indices), np.asarray(test_indices)])
    if combined.size != n_samples:
        return False
    if combined.min() < 0 or combined.max() >= n_samples:
        return False
    return np.unique(combined).size == n_samples


def load_create_split(
    data_dir: str, test_ratio: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Load or create a fixed stratified train/test partition.

    Existing indices are reused only when they still form a complete partition.
    The default seed of 42 affects only split creation; once saved, every caller
    reuses the same indices regardless of its experiment seed.

    Args:
        data_dir: Root directory accepted by ``ImageFolder``.
        test_ratio: Fraction of samples assigned to the test partition.
        seed: Seed used when a new partition must be generated.

    Returns:
        Arrays containing training and test indices.
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


def get_transform(
    config: Mapping[str, Any],
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build the training and test preprocessing pipelines.

    All images become normalized 128×128 grayscale tensors. Training
    augmentation is selected through ``experiment_type``; test preprocessing is
    deterministic. ``small_aug`` applies 15° rotation and flipping;
    ``small_aug_reg`` adds random erasing before normalization;
    ``small_rotation`` uses 25° rotation; ``small_flip`` uses flipping; and
    ``small_erase`` uses random erasing. All other values receive no augmentation.

    Args:
        config: Experiment mapping containing ``experiment_type``.

    Returns:
        ``(training_transform, test_transform)``.
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


class TransformedSubset(Dataset):
    """Apply a transform lazily to samples from an existing dataset subset."""

    def __init__(
        self, subset: Dataset, transform: Callable[[Any], Any] | None = None
    ) -> None:
        """Store the wrapped subset and optional input transform."""
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Return one transformed input and its unchanged class label."""
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        """Return the number of samples in the wrapped subset."""
        return len(self.subset)


def load_datasets(
    config: Mapping[str, Any], seed: int
) -> tuple[DataLoader, DataLoader]:
    """Construct training and test loaders for one experiment run.

    Every experiment on a dataset shares its persisted train/test partition.
    Non-baseline experiments may use ``data_ratio`` to select a seed-specific,
    stratified fraction of that training partition. ``data_ratio`` is ignored
    when ``experiment_type`` is ``baseline``. The training loader uses the
    configured batch size and shuffles; the test loader is deterministic with a
    fixed batch size of 64.

    Args:
        config: Dataset, split, augmentation, and batch-size settings.
        seed: Seed for limited-training-data subsampling.

    Returns:
        Shuffled training and deterministic test loaders.
    """
    train_transform, test_transform = get_transform(config)

    train_indices, test_indices = load_create_split(
        data_dir = config['data_dir'],
        test_ratio = config['test_ratio']
    )

    # Split the untransformed source first so test images are never augmented.
    full_dataset = ImageFolder(root=config['data_dir'])

    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    test_dataset = TransformedSubset(test_subset, transform=test_transform)


    if config['experiment_type'] != 'baseline':
        if config['data_ratio'] < 1.0:
            # Resolve labels through the source ImageFolder for stratified sampling.
            train_labels = np.array([train_subset.dataset.targets[i] for i in train_subset.indices])

            subset_size = int(len(train_dataset) * config['data_ratio'])

            reduced_indices, _ = train_test_split(
                np.arange(len(train_dataset)),
                train_size=subset_size,
                stratify=train_labels,
                random_state=seed,
                shuffle=True
            )
            train_dataset = Subset(train_dataset, reduced_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader