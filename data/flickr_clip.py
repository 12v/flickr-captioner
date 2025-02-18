import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

from utils import device

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-base-patch32", pad_token="<|padding|>", unk_token="<|unknown|>"
)

clip_model.eval()
clip_text.eval()


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

        clip_text_outputs = clip_text(
            input_ids=input_tokens,
            attention_mask=input_padding_mask,
        )
        input_text_embeddings = clip_text_outputs.last_hidden_state.to(device)

        clip_image_inputs = clip_processor(images=photos, return_tensors="pt").to(
            device
        )

        image_embeddings = clip_model.get_image_features(**clip_image_inputs).to(device)

        return (
            image_embeddings,
            input_text_embeddings,
            output_tokens,
            input_padding_mask,
        )
