import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import Attention


class DecoderLayer(nn.Module):
    def __init__(self, d_model_decoder, num_heads, dropout_rate):
        super().__init__()
        self.masked_self_attention = Attention(
            query_dim=d_model_decoder,
            key_value_dim=d_model_decoder,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            causal_mask=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model_decoder, d_model_decoder * 4),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(d_model_decoder * 4, d_model_decoder),
        )
        self.norm1 = nn.LayerNorm(d_model_decoder)
        self.norm2 = nn.LayerNorm(d_model_decoder)
        self.dropout_rate = dropout_rate

    def forward(self, embeddings, padding_mask):
        x = self.norm1(embeddings)
        x, _ = self.masked_self_attention(
            x,
            x,
            key_padding_mask=padding_mask,
        )
        x = F.dropout(x, self.dropout_rate)
        attended_embeddings = embeddings + x
        x = self.norm2(attended_embeddings)
        x = self.feed_forward(x)
        x = F.dropout(x, self.dropout_rate)
        return attended_embeddings + x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model_decoder,
        decoder_length,
        num_decoder_layers,
        vocab_size,
        num_heads,
        padding_index,
        dropout_rate,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model_decoder)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model_decoder, num_heads, dropout_rate)
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model_decoder)
        self.output_layer = nn.Linear(d_model_decoder, vocab_size)
        self.padding_index = padding_index
        self.dropout_rate = dropout_rate

    def compute_loss(
        self,
        image_embedding,
        input_tokens,
        output_labels,
        input_padding_mask,
        image_padding_mask,
    ):
        x = self.forward(
            image_embedding, input_tokens, input_padding_mask, image_padding_mask
        )
        x = x.permute(0, 2, 1)
        return nn.CrossEntropyLoss(ignore_index=self.padding_index)(x, output_labels)

    def combine_embeddings(
        self, image_embedding, text_embeddings, input_padding_mask, image_padding_mask
    ):
        combined_embeddings = torch.cat([image_embedding, text_embeddings], dim=1)

        combined_padding_mask = torch.cat(
            [image_padding_mask, input_padding_mask], dim=1
        )
        combined_padding_mask = combined_padding_mask == 0

        return combined_embeddings, combined_padding_mask

    def forward(
        self, image_embedding, input_tokens, text_padding_mask, image_padding_mask
    ):
        x = self.embedding(input_tokens)
        x, combined_padding_mask = self.combine_embeddings(
            image_embedding, x, text_padding_mask, image_padding_mask
        )
        x = F.dropout(x, self.dropout_rate)

        for layer in self.decoder_layers:
            x = layer(x, combined_padding_mask)

        x = self.norm(x)
        x = self.output_layer(x)

        return x[:, image_embedding.shape[1] :, :]
