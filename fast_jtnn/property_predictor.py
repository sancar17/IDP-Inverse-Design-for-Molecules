import torch
import torch.nn as nn
from fast_jtnn import *

class PropertyPredictor(nn.Module):
    def __init__(self, input_size):
        super(PropertyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 67),
            nn.BatchNorm1d(67),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(67, 67),
            nn.BatchNorm1d(67),
            nn.PReLU(), 
            nn.Dropout(0.15),
            nn.Linear(67, 6)  # 3 properties with mean and log_std for each
        )

    def forward(self, x):
        return self.model(x)
    
class PropertyPredictor1(nn.Module):
    def __init__(self, input_size):
        super(PropertyPredictor1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 67),
            nn.BatchNorm1d(67),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(67, 67),
            nn.BatchNorm1d(67),
            nn.PReLU(), 
            nn.Dropout(0.15),
            nn.Linear(67, 2)  # 3 properties with mean and log_std for each
        )

    def forward(self, x):
        return self.model(x)

def nll_gaussian1(y_pred, y_true):
    means = y_pred[:, 0]
    log_stds = y_pred[:, 1]
    stds = torch.exp(log_stds)
    loss = (log_stds + ((means - y_true) ** 2) / (stds)) / 2
    return loss.mean() 