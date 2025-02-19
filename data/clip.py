import torch
from transformers import CLIPModel, CLIPProcessor

from utils import device

pretrained_model = "openai/clip-vit-base-patch32"

clip_image_model = CLIPModel.from_pretrained(pretrained_model).to(device)
clip_processor = CLIPProcessor.from_pretrained(pretrained_model)


def get_image_embeddings(photos):
    with torch.no_grad():
        clip_image_inputs = clip_processor(images=photos, return_tensors="pt").to(
            device
        )

        clip_image_outputs = clip_image_model.get_image_features(**clip_image_inputs)
        return clip_image_outputs.to(device)
