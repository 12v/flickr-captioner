import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()

        # Create a matrix of shape (max_len, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(
            1
        )  # Shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )  # Shape: (d_model/2,)

        # Calculate the positional encodings
        pe = torch.zeros(length, d_model)  # Shape: (max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer(
            "pe", pe
        )  # Register as a buffer to avoid being a model parameter

    def forward(self, x):
        # Add positional encoding to the input embeddings
        # x should have shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # Add positional encodings
        return x
