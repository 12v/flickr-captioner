import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, length, d_model))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]
