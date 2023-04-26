from src.data.rainfall_dataset import RainfallDataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def get_dataset(csv_path, batch_size, train_window):
    """load rainfall dataset and returns train, val and test dataloaders

    Args:
        csv_path (_type_): path of the csc file which contains data
        batch_size (_type_): batch size for processing

    Returns:
        _type_: _description_
    """
    train_data = RainfallDataset(csv_file=csv_path, split='train', train_window= train_window)
    val_data = RainfallDataset(csv_file=csv_path, split='val', train_window = train_window)
    test_data = RainfallDataset(csv_file=csv_path, split='test', train_window = train_window)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle= True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle= True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle= False)
    return train_loader, val_loader, test_loader
