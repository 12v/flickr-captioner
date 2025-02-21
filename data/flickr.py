import os
import pickle

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from data.bert import get_text_tokens
from data.pickler import fast_seek, index_pickle
from data.pixtral import calculate_embeddings, combine_and_pad_embeddings

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)


length = len(ds)
train_length = int(length * 0.8)
test_length = length - train_length

train_ds = ds.select(range(train_length))
test_ds = ds.select(range(train_length, length))


embedding_path = os.path.join(script_dir, "flickr30k_embedded_pixtral.pkl")
offsets_path = os.path.join(script_dir, "offsets.pkl")

if not os.path.exists(embedding_path):
    print("Computing image embeddings")
    pickle_file = os.path.join(script_dir, "flickr30k_embedded_pixtral.pkl")
    with open(pickle_file, "ab") as f:
        for i, photo in enumerate([item["image"] for item in train_ds]):
            if i % 100 == 0:
                print(f"Processing image {i} of {len(train_ds)}")
            embeddings = calculate_embeddings(photo)
            pickle.dump(embeddings, f)
        for i, photo in enumerate([item["image"] for item in test_ds]):
            if i % 100 == 0:
                print(f"Processing image {i} of {len(test_ds)}")
            embeddings = calculate_embeddings(photo)
            pickle.dump(embeddings, f)
    print("Saved embeddings to", embedding_path)

if not os.path.exists(offsets_path):
    print("Computing offsets")
    offsets = index_pickle(embedding_path)

    with open(offsets_path, "wb") as f:
        pickle.dump(offsets, f)

with open(offsets_path, "rb") as f:
    print("Loading offsets")
    offsets = pickle.load(f)

train_offsets = offsets[:train_length]
test_offsets = offsets[train_length:-1]


class FlickrDataset(Dataset):
    def __init__(self, dataset, offsets, caption_length):
        self.dataset = dataset
        self.offsets = offsets
        self.caption_length = caption_length

    def __len__(self):
        return len(self.dataset) * 5

    def __getitem__(self, idx):
        photo_id = idx // 5
        caption_id = idx % 5

        image_embedding = fast_seek(embedding_path, self.offsets, photo_id)
        caption = self.dataset[photo_id]["caption"][caption_id]

        return image_embedding, caption


def _collate_fn(batch, caption_length):
    image_embeddings = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    with torch.no_grad():
        input_tokens, output_tokens, input_padding_mask = get_text_tokens(
            captions, caption_length
        )

        photo_embeddings, image_padding_mask = combine_and_pad_embeddings(
            image_embeddings
        )

        return (
            photo_embeddings,
            input_tokens,
            output_tokens,
            input_padding_mask,
            image_padding_mask,
        )
