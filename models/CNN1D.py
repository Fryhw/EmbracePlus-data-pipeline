import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNDenoiser1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 1, 5, padding=2)
        )

    def forward(self, x):
        return self.net(x)
