import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTMDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, 1]
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.transpose(1, 2)  # back to [B, 1, T]
