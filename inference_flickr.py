import torch
import torch.nn.functional as F

from data.clip_caption_dataset import (
    finish_token,
    inference_generator,
    padding_token,
    start_token,
    test_ds,
    tokenizer,
    vocab_size,
)
from data.visualization import visualize_image
from model.decoder import Decoder
from params_flickr import (
    d_model_decoder,
    decoder_length,
    num_decoder_layers,
    num_heads,
)
from utils import device

model = Decoder(
    d_model_decoder=d_model_decoder,
    decoder_length=decoder_length,
    vocab_size=vocab_size,
    num_decoder_layers=num_decoder_layers,
    num_heads=num_heads,
    padding_index=tokenizer.get_id_for_token(padding_token),
)


model.load_state_dict(torch.load("weights/decoder_0.pth", map_location=device))

# count number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.to(device)

model.eval()
with torch.no_grad():
    for embedding, caption, photo in inference_generator(test_ds):
        input_tokens = [tokenizer.get_id_for_token(start_token)]
        output_tokens = []

        for i in range(decoder_length - 1):
            tokens = torch.stack((torch.tensor(input_tokens),))
            output = model(
                embedding.unsqueeze(0).to(device),
                tokens.to(device),
                torch.ones_like(tokens).to(device),
            )

            softmax_output = F.softmax(output[0][0][1:], dim=-1)
            output_token = torch.multinomial(softmax_output, 1).item()

            input_tokens.append(output_token)
            output_tokens.append(output_token)

            print(output_tokens, end="\r")
            output_text = tokenizer.decode(output_tokens)

            if output_token == tokenizer.get_id_for_token(finish_token):
                break

        print("\n")
        print(output_text)
        visualize_image(photo)
