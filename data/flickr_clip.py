import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

from utils import device

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

pretrained_model = "openai/clip-vit-base-patch32"

clip_image_model = CLIPModel.from_pretrained(pretrained_model).to(device)
clip_processor = CLIPProcessor.from_pretrained(pretrained_model)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for param in clip_image_model.parameters():
    param.requires_grad = False

clip_image_model.eval()

length = len(ds)
train_length = int(length * 0.8)
test_length = length - train_length

train_ds = ds.select(range(train_length))
test_ds = ds.select(range(train_length, length))


def get_image_embeddings(photos):
    with torch.no_grad():
        clip_image_inputs = clip_processor(images=photos, return_tensors="pt").to(
            device
        )

        clip_image_outputs = clip_image_model.get_image_features(**clip_image_inputs)
        return clip_image_outputs.to(device)


def cache_image_embeddings(photos, batch_size=256):
    embeddings = []
    for i in range(0, len(photos), batch_size):
        print(i)
        batch = photos[i : i + batch_size]
        with torch.no_grad():
            image_embeddings = get_image_embeddings(batch)
            embeddings.append(image_embeddings.detach().clone().cpu())
    return torch.cat(embeddings, dim=0)


embedding_path = os.path.join(script_dir, "flickr30k_embedded.pt")

if not os.path.exists(embedding_path):
    print("Computing image embeddings")
    train_embeddings = cache_image_embeddings([item["image"] for item in train_ds])
    test_embeddings = cache_image_embeddings([item["image"] for item in test_ds])

    torch.save(
        {"train_embeddings": train_embeddings, "test_embeddings": test_embeddings},
        embedding_path,
    )
    print("Saved embeddings to", embedding_path)
else:
    print("Loading pre-computed embeddings")
    data = torch.load(embedding_path)
    train_embeddings = data["train_embeddings"]
    test_embeddings = data["test_embeddings"]


class FlickrClipDataset(Dataset):
    def __init__(self, dataset, embeddings, caption_length):
        self.dataset = dataset
        self.caption_length = caption_length
        self.embeddings = embeddings

    def __len__(self):
        return len(self.dataset) * 5

    def __getitem__(self, idx):
        photo_id = idx // 5
        caption_id = idx % 5

        image_embedding = self.embeddings[photo_id]
        caption = self.dataset[photo_id]["caption"][caption_id]

        return image_embedding, caption


def _collate_fn(batch, caption_length):
    photos = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    with torch.no_grad():
        input_tokens, output_tokens, input_padding_mask = get_text_tokens(
            captions, caption_length
        )

        return (
            torch.stack(photos),
            input_tokens,
            output_tokens,
            input_padding_mask,
        )


def get_text_tokens(captions, caption_length):
    with torch.no_grad():
        clip_text_inputs = bert_tokenizer(
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

        return (
            input_tokens,
            output_tokens,
            input_padding_mask,
        )
