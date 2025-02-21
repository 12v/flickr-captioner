import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from inference_flickr import inference

app = FastAPI()


async def processing_image(file: UploadFile):
    output_string = ""

    for latest in inference(file):
        yield latest[len(output_string) :]
        output_string = latest


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return StreamingResponse(processing_image(image), media_type="text/plain")
