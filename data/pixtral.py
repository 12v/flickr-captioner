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


def get_image_embeddings(photo):
    with torch.no_grad():
        # outputs = []
        # masks = []
        # for i, photo in enumerate(photos):
        #     print(f"Processing image {i} of {len(photos)}")
        pixtral_image_inputs = processor(images=photo, return_tensors="pt").to(device)

        pixtral_image_outputs = model.forward(**pixtral_image_inputs)

        embeddings = pixtral_image_outputs.last_hidden_state
        print(embeddings.dtype)
        return embeddings.clone().cpu().detach()

    #     outputs.append(embeddings.clone().cpu().detach().half())
    # return outputs

    # max_length = max(output.shape[1] for output in outputs)

    # padded_outputs = []
    # for i, output in enumerate(outputs):
    #     if i % 100 == 0:
    #         print(f"Padding image {i} of {len(outputs)}")
    #     padded_output = torch.cat(
    #         [
    #             output,
    #             torch.zeros(
    #                 output.shape[0],
    #                 max_length - output.shape[1],
    #                 output.shape[2],
    #                 device=device,
    #             ),
    #         ],
    #         dim=1,
    #     )
    #     mask = torch.ones(
    #         output.shape[0], max_length, dtype=torch.long, device=device
    #     )
    #     mask[:, output.shape[1] :] = 0
    #     masks.append(mask)
    #     padded_outputs.append(padded_output)

    # stacked_outputs = torch.stack(padded_outputs)
    # stacked_masks = torch.stack(masks)
    # stacked_outputs = stacked_outputs.squeeze(1)
    # return stacked_outputs.to(device), stacked_masks.to(device)
