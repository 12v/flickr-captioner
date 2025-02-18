import multiprocessing as mp
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DummyWandb, device

if torch.cuda.is_available():
    import wandb
else:
    wandb = DummyWandb()

from data.flickr_clip import (
    FlickrClipDataset,
    _collate_fn,
    clip_tokenizer,
    test_ds,
    train_ds,
)
from model.decoder import Decoder
from params_flickr import (
    d_image_embeddings,
    d_model_decoder,
    decoder_length,
    num_decoder_layers,
    num_heads,
)

script_dir = os.path.dirname(os.path.abspath(__file__))


def train():
    num_epochs = 10
    batch_size = 128 if torch.cuda.is_available() else 100
    learning_rate = 3e-3 if torch.cuda.is_available() else 1e-3
    num_workers = 4 if torch.cuda.is_available() else 0
    persistent_workers = True if num_workers > 0 else False

    partial_collate_fn = partial(_collate_fn, caption_length=decoder_length - 1)

    training_dataset = FlickrClipDataset(train_ds, caption_length=decoder_length - 1)
    validation_dataset = FlickrClipDataset(test_ds, caption_length=decoder_length - 1)

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=partial_collate_fn,
        persistent_workers=persistent_workers,
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=partial_collate_fn,
        persistent_workers=persistent_workers,
    )

    decoder = Decoder(
        d_model_decoder=d_model_decoder,
        d_image_embeddings=d_image_embeddings,
        decoder_length=decoder_length,
        num_decoder_layers=num_decoder_layers,
        vocab_size=clip_tokenizer.vocab_size,
        num_heads=num_heads,
        padding_index=clip_tokenizer.pad_token_id,
    )

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

    decoder = decoder.to(device)

    wandb.init(
        project="flickr-captioning-clip-tokenizer",
        config={
            "d_model_decoder": d_model_decoder,
            "decoder_length": decoder_length,
            "num_decoder_layers": num_decoder_layers,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        },
    )

    for epoch in range(num_epochs):
        decoder.train()
        batch_losses = []
        train_loop = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(train_dataloader),
        )

        for (
            image_embedding,
            input_text_embeddings,
            output_tokens,
            padding_mask,
        ) in train_loop:
            image_embedding = image_embedding.to(device)
            input_text_embeddings = input_text_embeddings.to(device)
            output_tokens = output_tokens.to(device)
            padding_mask = padding_mask.to(device)

            loss = decoder.compute_loss(
                image_embedding,
                input_text_embeddings,
                output_tokens,
                padding_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            train_loop.set_postfix(loss=f"{sum(batch_losses) / len(batch_losses):.4f}")
            wandb.log({"loss": loss.item()})

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {sum(batch_losses) / len(batch_losses):.4f}"
        )
        os.makedirs(os.path.join(script_dir, "weights"), exist_ok=True)
        torch.save(
            decoder.state_dict(),
            os.path.join(script_dir, f"weights/decoder_{epoch}.pth"),
        )
        wandb.save(os.path.join(script_dir, f"weights/decoder_{epoch}.pth"))

        val_losses = []
        for (
            image_embedding,
            input_text_embeddings,
            output_tokens,
            padding_mask,
        ) in val_dataloader:
            decoder.eval()
            image_embedding = image_embedding.to(device)
            input_text_embeddings = input_text_embeddings.to(device)
            output_tokens = output_tokens.to(device)
            padding_mask = padding_mask.to(device)

            with torch.no_grad():
                loss = decoder.compute_loss(
                    image_embedding,
                    input_text_embeddings,
                    output_tokens,
                    padding_mask,
                )
            val_losses.append(loss.item())

        wandb.log(
            {
                "val_loss": sum(val_losses) / len(val_losses),
                "epoch_loss": sum(batch_losses) / len(batch_losses),
                "epoch": epoch + 1,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()
