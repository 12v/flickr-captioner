import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.clip_caption_dataset import Flickr30kDataset, train_ds, vocab_size
from model.decoder import Decoder
from params_flickr import (
    d_model_decoder,
    decoder_length,
    num_decoder_layers,
    num_heads,
)
from utils import device

script_dir = os.path.dirname(os.path.abspath(__file__))


def train():
    num_epochs = 10
    batch_size = 512 if device == "cuda" else 512
    learning_rate = 1e-3 if device == "cuda" else 1e-3
    num_workers = 4 if device == "cuda" else 2

    train_dataloader = DataLoader(
        Flickr30kDataset(train_ds, caption_length=decoder_length - 1),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # val_dataloader = DataLoader(
    #     Flickr30kDataset(test_ds),
    #     batch_size=batch_size,
    # )

    decoder = Decoder(
        d_model_decoder=d_model_decoder,
        decoder_length=decoder_length,
        num_decoder_layers=num_decoder_layers,
        vocab_size=vocab_size,
        num_heads=num_heads,
    )
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

    decoder = decoder.to(device)

    for epoch in range(num_epochs):
        decoder.train()
        batch_losses = []
        train_loop = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(train_ds) * 5 // batch_size,
        )

        for (
            image_embedding,
            input_caption,
            output_caption,
            padding_mask,
        ) in train_loop:
            image_embedding = image_embedding.to(device)
            input_caption = input_caption.to(device)
            output_caption = output_caption.to(device)
            padding_mask = padding_mask.to(device)

            loss = decoder.compute_loss(
                image_embedding, input_caption, output_caption, padding_mask
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            train_loop.set_postfix(loss=f"{sum(batch_losses) / len(batch_losses):.4f}")

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {sum(batch_losses) / len(batch_losses):.4f}"
        )
        torch.save(
            decoder.state_dict(),
            os.path.join(script_dir, f"weights/decoder_{epoch}.pth"),
        )

    return decoder


if __name__ == "__main__":
    train()
