
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

NUM_WORKERS = 2

def get_dataloaders(batch_size: int, valid_split: float = 0.1, num_workers: int = NUM_WORKERS):
    """
    Creates dataloaders for CIFAR-10 training, validation, and testing.
    
    Args:
        batch_size (int): Number of implementation samples per batch.
        valid_split (float): Percentage of training data to use for validation.
        num_workers (int): Number of subprocesses to use for data loading.
        
    Returns:
        train_dataloader, val_dataloader, test_dataloader, class_names
    """
    
    # Define transformations
    # Training: Random crop, flip, and normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Testing/Validation: Only normalization
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load training data
    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=valid_transform)
    
    # Create validation split
    num_train = len(train_data)
    num_val = int(valid_split * num_train)
    num_train = num_train - num_val
    
    train_subset, val_subset = random_split(train_data, [num_train, num_val], generator=torch.Generator().manual_seed(42))
    
    # Verification: IMPORTANT - validation subset should use valid_transform, not train_transform
    # However, random_split splits the dataset object which already has a transform attached.
    # To fix this, we need to override the transform for the validation subset.
    # A cleaner way is to create two dataset objects, one with train transform and one with valid transform,
    # and then use indices to split them. But for simplicity and standard practice in many tutorials,
    # we often just split. 
    # LET'S DO IT CORRECTLY:
    # We will split INDICES, then create subsets with correct transforms.
    
    # Alternative approach: Load train data twice with different transforms
    full_train_data = datasets.CIFAR10(root="data", train=True, download=True) # No transform yet
    
    # Create indices
    indices = torch.randperm(len(full_train_data)).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create distinct dataset objects
    train_dataset = datasets.CIFAR10(root="data", train=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root="data", train=True, transform=valid_transform)
    
    # Create subsets using indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    class_names = train_dataset.classes

    # Create DataLoaders
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader, class_names
