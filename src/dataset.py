import os
import torch
from torch.utils.data import Subset, random_split, DataLoader
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms


def get_transform(config):
    """
    returns a dataset preprocessor for training
    datasets as train_transform, additionally applies data 
    augmentation based on configuration
    returns a fixed dataset preprocessor for testing dataset
    as test_transform, same for all 4 experiments
    
    """

    if config['data_augmentation']:
        train_transform =  transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()  
    ]  
)
    elif config['data_augmentation'] and config['regularization']:
        train_transform =  transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.RandomRotation(30),   
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p = 0.15)
    ]  
)
    else:
        train_transform =  transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]  
)
    
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
        
    ]
)
    return train_transform, test_transform
        
        
def load_datasets(config):
    """
    returns a test dataloader and a train dataloader
    test dataloader is fixed for all experiments,
    train dataloader will vary in data ratio, data
    transformation, and batch size based on configs
    """
    
    train_transform, test_transform = get_transform(config)
    full_train_dataset = ImageFolder(root=config['data_dir'], transform=train_transform)
    full_test_dataset  = ImageFolder(root=config['data_dir'], transform=test_transform)
    

    test_ratio = config['test_ratio']
    test_size = int(len(full_train_dataset) * test_ratio)
    train_size = len(full_train_dataset) - test_size


    np.random.seed(42)
    indices = np.random.permutation(len(full_train_dataset))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # split dataset for training and testing
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)
    
    # for experiments working with subset of data
    if config['data_ratio'] < 1.0:
        subset_size = int(len(train_dataset) * config['data_ratio'])
        reduced_indices = np.random.choice(len(train_dataset), subset_size, replace = False)
        train_dataset = Subset(train_dataset, reduced_indices)
  

    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle= True) 
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False) # this test loader is fixed for all experiment
    return train_loader, test_loader