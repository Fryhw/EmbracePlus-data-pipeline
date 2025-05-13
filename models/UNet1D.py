import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.time_embed = nn.Sequential(
        nn.Linear(1, base_channels * 2),  # base_channels * 2 = 128
        nn.ReLU(),
        nn.Linear(base_channels * 2, base_channels * 2)  # must match encoder output channels
    )
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        # Time embedding
        t = t.view(-1, 1).float()  # [B, 1]
        t_embed = self.time_embed(t).unsqueeze(-1)  # [B, C, 1]
        x = self.encoder(x)
        x = x + t_embed  # Broadcast sur la longueur du signal
        x = self.decoder(x)
        return x
