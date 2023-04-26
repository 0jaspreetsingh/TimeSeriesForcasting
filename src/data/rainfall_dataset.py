import pandas as pd
import torch
from torch.utils.data import Dataset


class RainfallDataset(Dataset):
    def __init__(self, csv_file, split='train', test_size=0.2, val_size=0.1):
        """
        Args:
            csv_file (str): Path to the CSV file containing the data.
            split (str): One of 'train', 'val', or 'test'. Determines which split of the data to return.
            transform (callable, optional): Optional transform to be applied on a sample.
            test_size (float): Fraction of the data to be used for the test set.
            val_size (float): Fraction of the data to be used for the validation set.
        """

        # Load the CSV file into a Pandas dataframe
        self.df = pd.read_csv(csv_file)

        # Split the data into training, validation, and test sets
        if split == 'train':
            self.df = self.df[:-int(len(self.df)*(test_size+val_size))]
        elif split == 'val':
            self.df = self.df[-int(len(self.df)*val_size)-int(len(self.df)
                                                              * (test_size)): -int(len(self.df)*test_size)]
        elif split == 'test':
            self.df = self.df[-int(len(self.df)*test_size):]
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        # Convert the dataframe to a PyTorch tensor
        self.data = torch.tensor(self.df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, 1:]
        return sample
