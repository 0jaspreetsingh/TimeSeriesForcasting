import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, device, patience, log):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.log = log

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch_idx, (target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', leave=False):
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(target)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        return train_loss

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (target) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation', leave=False):
                target = target.to(self.device)
                output = self.model(target)
                loss = self.criterion(output, target)
                val_loss += loss.item()
            val_loss /= len(val_loader)
        return val_loss

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (target) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing', leave=False):
                target = target.to(self.device)
                output = self.model(target)
                loss = self.criterion(output, target)
                test_loss += loss.item()
            test_loss /= len(test_loader)
        return test_loss
    
    def run(self, epochs, train_loader, test_loader, val_loader):

        best_loss = np.inf
        no_improve = 0
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            val_loss = self.validate(val_loader)
            self.log.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(f = 'out/model.pt' , obj= self.model)
                no_improve = 0
            else:
                no_improve += 1
                if self.patience and no_improve > self.patience:
                    self.log.info(f"Stopping early at epoch {epoch+1} with no improvement for {no_improve} epochs.")
                    break
        test_loss = self.test(test_loader)
        self.log.info(f"Test Loss: {test_loss:.4f}")