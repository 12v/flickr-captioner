import torch
import torch.nn as nn

from model.attention import Attention
from model.positional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    def __init__(self, d_model_decoder, num_heads):
        super().__init__()
        self.masked_self_attention = Attention(
            query_dim=d_model_decoder,
            key_value_dim=d_model_decoder,
            num_heads=num_heads,
            causal_mask=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model_decoder, d_model_decoder * 4),
            nn.ReLU(),
            nn.Linear(d_model_decoder * 4, d_model_decoder),
        )
        self.norm1 = nn.LayerNorm(d_model_decoder)
        self.norm2 = nn.LayerNorm(d_model_decoder)

    def forward(self, embeddings, padding_mask):
        x = self.norm1(embeddings)
        attention, _ = self.masked_self_attention(embeddings, embeddings, padding_mask)
        x = self.norm2(x + attention)
        return self.feed_forward(x)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model_decoder,
        decoder_length,
        num_decoder_layers,
        vocab_size,
        num_heads,
    ):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, d_model_decoder)
        self.positional_encoder = PositionalEncoder(d_model_decoder, decoder_length)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model_decoder, num_heads)
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model_decoder)
        self.output_layer = nn.Linear(d_model_decoder, vocab_size)

    def compute_loss(self, image_embedding, input_labels, output_labels, padding_mask):
        x = self.forward(image_embedding, input_labels, padding_mask)
        x = x[:, 1:, :]
        x = torch.permute(x, (0, 2, 1))

        loss = nn.CrossEntropyLoss(reduction="none")(x, output_labels)

        loss = loss * padding_mask

        masked_loss_sum = loss.sum()
        num_non_padding = padding_mask.sum()
        final_loss = masked_loss_sum / num_non_padding

        return final_loss

    def forward(self, image_embedding, input_labels, padding_mask):
        label_embeddings = self.embedder(input_labels)
        image_embedding = image_embedding.unsqueeze(1)

        combined_embeddings = torch.cat([image_embedding, label_embeddings], dim=1)
        x = self.positional_encoder(combined_embeddings)

        full_padding_mask = torch.cat(
            [torch.ones_like(image_embedding)[..., 0], padding_mask], dim=1
        )

        for layer in self.decoder_layers:
            x = layer(x, full_padding_mask)

        x = self.norm(x)
        return self.output_layer(x)
