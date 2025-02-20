import torch
from transformers import PixtralImageProcessor, PixtralVisionConfig, PixtralVisionModel

from utils import device

processor = PixtralImageProcessor.from_pretrained("mistral-community/pixtral-12b")

config = PixtralVisionConfig.from_pretrained("mistral-community/pixtral-12b")

model = PixtralVisionModel(config).to(device)

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
