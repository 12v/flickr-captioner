import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_value_dim,
        num_heads,
        dropout_rate,
        causal_mask=False,
    ):
        super().__init__()
        self.causal_mask = causal_mask
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.query_weights = nn.Linear(query_dim, key_value_dim)  # Query
        self.key_weights = nn.Linear(key_value_dim, key_value_dim)  # Key
        self.value_weights = nn.Linear(key_value_dim, key_value_dim)  # Value

        self.output_layer = nn.Linear(key_value_dim, query_dim)

        self.dropout_rate = dropout_rate

        assert key_value_dim % num_heads == 0, (
            f"key_value_dim must be divisible by num_heads, got key_value_dim={key_value_dim} and num_heads={num_heads}"
        )

        self.head_dim = key_value_dim // num_heads

    def calculate_attention(self, query, key, value, key_padding_mask):
        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            key.shape[0], key.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim)
        )
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            expanded_key_padding_mask = key_padding_mask.expand(
                -1, attention.shape[1], -1, -1
            )
            attention = attention.masked_fill(expanded_key_padding_mask, float("-inf"))

        if self.causal_mask:
            attention = attention.masked_fill(
                torch.triu(torch.ones_like(attention), diagonal=1) == 1,
                float("-inf"),
            )
        raw_attention = torch.softmax(attention, dim=-1)
        dropped_out_raw_attention = F.dropout(raw_attention, self.dropout_rate)
        weighted_attention = torch.matmul(dropped_out_raw_attention, value)
        weighted_attention = (
            weighted_attention.transpose(1, 2)
            .contiguous()
            .view(
                weighted_attention.shape[0],
                weighted_attention.shape[2],
                -1,
            )
        )
        return weighted_attention, raw_attention

    def forward(self, query_states, key_value_states, key_padding_mask=None):
        query = self.query_weights(query_states)
        key = self.key_weights(key_value_states)
        value = self.value_weights(key_value_states)

        attention, raw_attention = self.calculate_attention(
            query, key, value, key_padding_mask
        )
        output = self.output_layer(attention)
        return output, raw_attention
