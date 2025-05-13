import torch
import torch.nn as nn
import torch.nn.functional as F
class DilatedCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, dilation=4, padding=4)
        )

    def forward(self, x):
        return self.layers(x)
