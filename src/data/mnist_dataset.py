# src/data/mnist_dataset.py

from torchvision import datasets, transforms    # dataset & transforms
from torch.utils.data import DataLoader, random_split  # data utilities

def get_mnist_loaders(data_dir: str, batch_size: int, val_split: float = 0.1):
    """
    Build DataLoaders for MNIST with augmentation & validation split.
    Args:
        data_dir: path to store/download data
        batch_size: batch size
        val_split: fraction of train → validation
    Returns:
        train_loader, val_loader, test_loader
    """
    # Compose augmentations
    transform = transforms.Compose([
        transforms.RandomRotation(10),          # ±10° rotation
        transforms.RandomCrop(28, padding=4),  # random crop w/ padding
        transforms.ToTensor(),                 # to PyTorch tensor
    ])

    # Download & wrap full training set
    full_train = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    total   = len(full_train)                 # total samples
    val_len = int(total * val_split)          # val set size
    train_len = total - val_len               # train set size

    # Split into train / val
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False)

    # Test set (no augment)
    test_ds = datasets.MNIST(
        root=data_dir, train=False, download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader
