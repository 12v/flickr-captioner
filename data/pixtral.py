import os

import requests
import safetensors.torch
import torch
from transformers import (
    PixtralImageProcessorFast,
    PixtralVisionConfig,
    PixtralVisionModel,
)

from utils import device

script_dir = os.path.dirname(os.path.abspath(__file__))
download_path = "https://huggingface.co/12v12v/p12b_vision_tower/resolve/main/vision_tower.safetensors"

output_path = os.path.join(script_dir, "pixtral_weights")
output_file = os.path.join(output_path, "vision_tower.safetensors")


def download_safetensors():
    os.makedirs(output_path, exist_ok=True)
    response = requests.get(download_path)
    with open(output_file, "wb") as f:
        f.write(response.content)


if not os.path.exists(output_file):
    print("Downloading Pixtral 12B weights")
    download_safetensors()

print("Loading Pixtral 12B weights")
weights = safetensors.torch.load_file(output_file)

processor = PixtralImageProcessorFast.from_pretrained("mistral-community/pixtral-12b")

config = PixtralVisionConfig.from_pretrained("mistral-community/pixtral-12b")

model = PixtralVisionModel(config).to(device)

model.load_state_dict(weights)

for param in model.parameters():
    param.requires_grad = False

model.eval()


def calculate_embeddings(photo):
    with torch.no_grad():
        pixtral_image_inputs = processor(images=photo, return_tensors="pt").to(device)

        pixtral_image_outputs = model.forward(**pixtral_image_inputs)

        embeddings = pixtral_image_outputs.last_hidden_state
        print(embeddings.dtype)
        return embeddings.clone().cpu().detach()


def combine_and_pad_embeddings(image_embeddings):
    max_length = max(embedding.shape[1] for embedding in image_embeddings)

    padded_embeddings = []
    masks = []

    fully_padded_embedding = torch.zeros(
        image_embeddings[0].shape[0],
        max_length,
        image_embeddings[0].shape[2],
        device=device,
    )

    for embedding in image_embeddings:
        padded_embedding = fully_padded_embedding.clone()
        padded_embedding[:, : embedding.shape[1]] = embedding
        mask = torch.ones(
            embedding.shape[0], max_length, dtype=torch.long, device=device
        )
        mask[:, embedding.shape[1] :] = 0
        masks.append(mask)
        padded_embeddings.append(padded_embedding)

    stacked_embeddings = torch.stack(padded_embeddings)
    stacked_masks = torch.stack(masks)
    print("stacked", stacked_embeddings.shape)
    print("stacked masks", stacked_masks.shape)
    stacked_embeddings = stacked_embeddings.squeeze(1)

    print("squeezed", stacked_embeddings.shape)
    return stacked_embeddings.to(device), stacked_masks.to(device)
