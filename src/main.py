import argparse
import configparser
import torch
import torch.optim as optim
import logging
from datetime import datetime
import math
import os
from src.model.lstm_model import LSTMModel

from src.data.loader import get_dataset
from src.trainer import Trainer


def get_parser():
    """get parser for runtime training arguments."""
    parser = argparse.ArgumentParser(description='TimeSeries Forcasting')

    parser.add_argument(
        '--notes',
        type=str,
        default=None,
        help='Notes about experiment')
    parser.add_argument(
        '--env',
        type=str,
        default='development',
        help='For loading congifuration, possible val: development or production')

    return parser


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = get_parser()
    args = parser.parse_args()

    script_start_time = datetime.now().timestamp()
    timestamp = f'{math.trunc(script_start_time)}'
    log.info(f'Build: {timestamp}, Started with parameters: {args}')
    config = configparser.ConfigParser()
    config.read(f'config/{args.env}.ini')

    # Loading dataset
    train_loader, val_loader, test_loader = get_dataset(csv_path=config['DEFAULT']['DataSetPath'], batch_size=config['DEFAULT'].getint('BatchSize'))
    model = LSTMModel(input_dim = 1, hidden_dim = 12, output_dim = 1, num_layers = 4).to(device= device)
    optimizer = optim.Adam(model.parameters(), lr= config['DEFAULT'].getfloat('LearningRate'))
    trainer = Trainer( model = model, criterion = torch.nn.MSELoss(), optimizer = optimizer, device = device, patience = config['DEFAULT'].getint('Patience'), log = log)
    trainer.run(epochs = config['DEFAULT'].getint('MaxEpochs'), train_loader = train_loader, test_loader= test_loader, val_loader = val_loader)


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    log = logging.getLogger(__name__)
    main()
