import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """Long Short term memory based model for timeseries

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size , input_len = x.shape
        x = x.view(batch_size, input_len , 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device=x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
