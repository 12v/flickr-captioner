import os
import random

import sentencepiece as spm
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import CLIPModel, CLIPProcessor

from utils import device

script_dir = os.path.dirname(os.path.abspath(__file__))

vocab_size = 10000


def convert_image_to_embedding(image):
    image_inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(
        device
    )
    with torch.no_grad():
        image_features = clip_model.get_image_features(image_inputs["pixel_values"])

    return image_features.squeeze().detach().clone().cpu()


embedding_path = os.path.join(script_dir, "flickr30k_embedded.pt")

embeddings = []
captions = []
if not os.path.exists(embedding_path):
    print("Computing image embeddings")
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for param in clip_model.parameters():
        param.requires_grad = False
    for i in range(len(ds)):
        if i % 1000 == 0:
            print(f"Processing image {i} of {len(ds)}")
        photo = ds[i]["image"]
        inner_captions = ds[i]["caption"]
        image_features = convert_image_to_embedding(photo)

        embeddings.append(image_features.squeeze().detach().clone().cpu())
        captions.append(inner_captions)

    save_path = os.path.join(script_dir, "flickr30k_embedded.pt")
    torch.save({"embeddings": torch.stack(embeddings), "captions": captions}, save_path)
else:
    data = torch.load(embedding_path)
    embeddings = data["embeddings"]
    captions = data["captions"]


length = len(embeddings)
train_length = int(length * 0.8)
test_length = length - train_length


train_ds = list(zip(embeddings[:train_length], captions[:train_length]))
test_ds = list(zip(embeddings[train_length:], captions[train_length:]))

start_token = "<s>"
finish_token = "</s>"
padding_token = "<pad>"


class Flickr30kTokenizer:
    def __init__(self, captions):
        corpus_path = os.path.join(script_dir, "corpus.txt")
        model_path = os.path.join(script_dir, "flickr30k.model")

        if not os.path.exists(model_path):
            if not os.path.exists(corpus_path):
                self.create_corpus(captions, corpus_path)

            self.train_model(corpus_path, model_path)

        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def create_corpus(self, captions, path):
        with open(path, "w") as f:
            for quint in captions:
                for caption in quint:
                    f.write(caption + "\n")

    def train_model(self, corpus_path, model_path):
        with open(model_path, "wb") as f:
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_writer=f,
                vocab_size=vocab_size,
                control_symbols=[padding_token],
            )

    def decode(self, tokens):
        return self.sp.decode(tokens)

    def encode(self, caption, length):
        tokens = self.sp.encode(caption)
        tokens = tokens[: length - 1]

        padding_id = self.sp.piece_to_id(padding_token)

        input_encoding = [self.sp.piece_to_id(start_token)] + tokens
        output_encoding = tokens + [self.sp.piece_to_id(finish_token)]

        padded_input_encoding = input_encoding + [padding_id] * (
            length - len(input_encoding)
        )
        padded_output_encoding = output_encoding + [padding_id] * (
            length - len(output_encoding)
        )
        padding_mask = [1] * len(input_encoding) + [0] * (length - len(input_encoding))

        return padded_input_encoding, padded_output_encoding, padding_mask

    def get_id_for_token(self, token):
        return self.sp.piece_to_id(token)


tokenizer = Flickr30kTokenizer(captions)


def embedding_and_caption_generator(ds):
    options = []
    for i in range(len(ds)):
        for j in range(5):
            options.append((i, j))

    random.shuffle(options)

    for photo_index, caption_index in options:
        embedding = ds[photo_index][0]
        caption = ds[photo_index][1][caption_index]

        yield embedding, caption


def inference_generator(ds):
    flickr = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

    options = []
    for i in range(len(ds)):
        for j in range(5):
            options.append((i, j))

    random.shuffle(options)

    for photo_index, caption_index in options:
        embedding = ds[photo_index][0]
        caption = ds[photo_index][1][caption_index]
        photo = flickr[photo_index]["image"]

        yield embedding, caption, photo


class Flickr30kDataset(IterableDataset):
    def __init__(self, ds, caption_length):
        self.ds = ds
        self.caption_length = caption_length

    def __iter__(self):
        dataset = self.get_worker_ds(self.ds)
        generator = embedding_and_caption_generator(dataset)
        for photo, caption in generator:
            input_caption, output_caption, padding_mask = tokenizer.encode(
                caption, self.caption_length
            )

            yield (
                photo,
                torch.tensor(input_caption),
                torch.tensor(output_caption),
                torch.tensor(padding_mask),
            )

    def get_worker_ds(self, ds):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return ds
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        dataset_size = len(ds)
        per_worker = dataset_size // num_workers
        return ds[worker_id * per_worker : (worker_id + 1) * per_worker]
