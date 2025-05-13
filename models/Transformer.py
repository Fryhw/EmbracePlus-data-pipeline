import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerDenoiser(nn.Module):
    def __init__(self, seq_len=256):
        super().__init__()
        self.embedding = nn.Linear(1, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.output = nn.Linear(64, 1)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, 1]
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x.transpose(1, 2)  # [B, 1, T]
