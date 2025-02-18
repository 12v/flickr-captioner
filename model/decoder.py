import torch
import torch.nn as nn

from model.positional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    def __init__(self, d_model_decoder, num_heads):
        super().__init__()
        self.masked_self_attention = nn.MultiheadAttention(
            embed_dim=d_model_decoder,
            num_heads=num_heads,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model_decoder, d_model_decoder * 4),
            nn.GELU(),
            nn.Linear(d_model_decoder * 4, d_model_decoder),
        )
        self.norm1 = nn.LayerNorm(d_model_decoder)
        self.norm2 = nn.LayerNorm(d_model_decoder)

    def forward(self, embeddings, padding_mask):
        x = self.norm1(embeddings)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            embeddings.shape[1], device=embeddings.device
        )
        attention, _ = self.masked_self_attention(
            embeddings,
            embeddings,
            embeddings,
            is_causal=True,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask,
        )
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
        padding_index,
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoder(d_model_decoder, decoder_length)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model_decoder, num_heads)
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model_decoder)
        self.output_layer = nn.Linear(d_model_decoder, vocab_size)
        self.padding_index = padding_index

    def compute_loss(
        self,
        image_embedding,
        input_text_embeddings,
        output_labels,
        input_padding_mask,
    ):
        x = self.forward(image_embedding, input_text_embeddings, input_padding_mask)
        x = x[:, 1:, :]
        x = x.permute(0, 2, 1)

        return nn.CrossEntropyLoss(ignore_index=self.padding_index)(x, output_labels)

    def forward(self, image_embedding, input_text_embeddings, input_padding_mask):
        image_embedding = image_embedding.unsqueeze(1)
        combined_embeddings = torch.cat([image_embedding, input_text_embeddings], dim=1)

        image_embedding_mask = torch.ones_like(image_embedding)[:, :, 0]
        combined_padding_mask = torch.cat(
            [image_embedding_mask, input_padding_mask], dim=1
        )

        x = self.positional_encoder(combined_embeddings)

        for layer in self.decoder_layers:
            x = layer(x, combined_padding_mask)

        x = self.norm(x)
        return self.output_layer(x)
