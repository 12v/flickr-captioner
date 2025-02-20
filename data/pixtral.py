import os

import requests
import safetensors.torch
import torch
from transformers import PixtralImageProcessor, PixtralVisionConfig, PixtralVisionModel

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

processor = PixtralImageProcessor.from_pretrained("mistral-community/pixtral-12b")

config = PixtralVisionConfig.from_pretrained("mistral-community/pixtral-12b")

model = PixtralVisionModel(config).to(device)

model.load_state_dict(weights)

for param in model.parameters():
    param.requires_grad = False

model.eval()


def get_image_embeddings(photos):
    with torch.no_grad():
        outputs = []
        for photo in photos:
            pixtral_image_inputs = processor(images=photo, return_tensors="pt").to(
                device
            )

            pixtral_image_outputs = model.forward(**pixtral_image_inputs)

            pooled_embeddings = pixtral_image_outputs.last_hidden_state.max(dim=1)[0]
            outputs.append(pooled_embeddings)

        stacked_outputs = torch.stack(outputs)
        stacked_outputs = stacked_outputs.squeeze(1)
        return stacked_outputs.to(device)
