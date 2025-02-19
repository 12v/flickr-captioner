import torch
from transformers import AutoTokenizer

from utils import device

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


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
