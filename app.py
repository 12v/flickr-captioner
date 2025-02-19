import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

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

app = FastAPI()

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


async def processing_image(file: UploadFile):
    with torch.no_grad():
        image_embeddings = get_image_embeddings([file])
        input_tokens = [bert_tokenizer.cls_token_id]
        output_tokens = []
        output_string = ""

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

            print(output_tokens, end="\r")
            new_output_string = bert_tokenizer.decode(output_tokens)
            yield new_output_string[len(output_string) :]
            output_string = new_output_string


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return StreamingResponse(processing_image(image), media_type="text/plain")


@app.get("/")
async def root():
    return FileResponse("index.html")
