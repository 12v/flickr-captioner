import torch
import torch.nn.functional as F

from data.bert import bert_tokenizer
from data.clip import get_image_embeddings
from model.decoder import Decoder
from params_flickr import (
    d_model_decoder,
    decoder_length,
    dropout_rate,
    num_decoder_layers,
    num_heads,
)
from utils import device

model = Decoder(
    d_model_decoder=d_model_decoder,
    decoder_length=decoder_length,
    vocab_size=bert_tokenizer.vocab_size,
    num_decoder_layers=num_decoder_layers,
    num_heads=num_heads,
    padding_index=bert_tokenizer.pad_token_id,
    dropout_rate=dropout_rate,
)


model.load_state_dict(torch.load("weights/decoder_gpu.pth", map_location=device))
model.to(device)
model.eval()


def inference(image):
    with torch.no_grad():
        input_tokens = [bert_tokenizer.cls_token_id]
        output_tokens = []

        image_embeddings = get_image_embeddings([image])

        for i in range(decoder_length - 1):
            input_token_tensor = torch.tensor([input_tokens]).to(device)
            padding_mask = torch.ones_like(input_token_tensor).to(device)

            output = model(
                image_embeddings.to(device),
                input_token_tensor.to(device),
                padding_mask.to(device),
            )

            softmax_output = F.softmax(output[0][i + 1], dim=-1)
            output_token = torch.argmax(softmax_output).item()

            if output_token == bert_tokenizer.sep_token_id:
                break

            input_tokens.append(output_token)
            output_tokens.append(output_token)
            yield bert_tokenizer.decode(output_tokens)


if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt

    from data.flickr import test_ds

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    while True:
        random_index = random.randint(0, len(test_ds) - 1)
        image = test_ds[random_index]["image"]

        for chunk in inference(image):
            print(chunk, end="\r")
        print("\n")
        plt.imshow(image)
        plt.axis("off")
        plt.show()
