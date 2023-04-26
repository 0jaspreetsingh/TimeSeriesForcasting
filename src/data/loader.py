from src.data.rainfall_dataset import RainfallDataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def get_dataset(csv_path, batch_size):
    """load rainfall dataset and returns train, val and test dataloaders

    Args:
        csv_path (_type_): path of the csc file which contains data
        batch_size (_type_): batch size for processing

    Returns:
        _type_: _description_
    """
    train_data = RainfallDataset(csv_file=csv_path, split='train')
    val_data = RainfallDataset(csv_file=csv_path, split='val')
    test_data = RainfallDataset(csv_file=csv_path, split='test')

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader
