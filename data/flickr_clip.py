import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModel,
)

from utils import device

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

pretrained_model = "openai/clip-vit-base-patch32"

clip_image_model = CLIPVisionModel.from_pretrained(pretrained_model).to(device)
clip_processor = CLIPProcessor.from_pretrained(pretrained_model)
clip_text_model = CLIPTextModelWithProjection.from_pretrained(pretrained_model).to(
    device
)
clip_tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)

for param in clip_image_model.parameters():
    param.requires_grad = False

for param in clip_text_model.parameters():
    param.requires_grad = False

clip_image_model.eval()
clip_text_model.eval()


length = len(ds)
train_length = int(length * 0.8)
test_length = length - train_length

train_ds = ds.select(range(train_length))
test_ds = ds.select(range(train_length, length))


class FlickrClipDataset(Dataset):
    def __init__(self, dataset, caption_length):
        self.dataset = dataset
        self.caption_length = caption_length

    def __len__(self):
        return len(self.dataset) * 5

    def __getitem__(self, idx):
        photo_id = idx // 5
        caption_id = idx % 5

        photo = self.dataset[photo_id]["image"]
        caption = self.dataset[photo_id]["caption"][caption_id]

        return photo, caption


def _collate_fn(batch, caption_length):
    photos = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    with torch.no_grad():
        input_text_embeddings, output_tokens, input_padding_mask = get_text_embeddings(
            captions, caption_length
        )

        image_embeddings = get_image_embeddings(photos)

        return (
            image_embeddings,
            input_text_embeddings,
            output_tokens,
            input_padding_mask,
        )


def get_text_embeddings(captions, caption_length):
    with torch.no_grad():
        clip_text_inputs = clip_tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=caption_length,
        ).to(device)

        input_padding_mask = clip_text_inputs.attention_mask
        input_tokens = clip_text_inputs.input_ids

        output_tokens = input_tokens[:, 1:]
        input_tokens = input_tokens[:, :-1]
        input_padding_mask = input_padding_mask[:, :-1]

        clip_text_outputs = get_text_embeddings_from_token_ids(
            input_tokens, input_padding_mask
        )

        return (
            clip_text_outputs,
            output_tokens,
            input_padding_mask,
        )


def get_text_embeddings_from_token_ids(token_ids, padding_mask):
    with torch.no_grad():
        clip_text_outputs = clip_text_model(
            input_ids=token_ids,
            attention_mask=padding_mask,
        )
        last_hidden_state = clip_text_outputs.last_hidden_state
        return last_hidden_state.to(device)


def get_image_embeddings(photos):
    with torch.no_grad():
        clip_image_inputs = clip_processor(images=photos, return_tensors="pt").to(
            device
        )

        clip_image_outputs = clip_image_model(**clip_image_inputs)
        pooler_output = clip_image_outputs.pooler_output
        return pooler_output.to(device)
