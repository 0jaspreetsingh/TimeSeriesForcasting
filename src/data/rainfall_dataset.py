import pandas as pd
import torch
from torch.utils.data import Dataset


class RainfallDataset(Dataset):
    def __init__(self, csv_file, split='train', test_size=0.2, val_size=0.1, train_window = 12):
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
        
        data = torch.tensor(self.df.iloc[:,1:].to_numpy(dtype='float32').flatten())
        input_seq = []
        L = len(data)
        for i in range(L-train_window):
            train_seq = data[i:i+train_window]
            label = data[i+train_window:i+train_window+1]
            input_seq.append((train_seq ,label))

        self.data = input_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        train_seq , label = self.data[idx]
        return train_seq , label 
