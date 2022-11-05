import io

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel
from starlette.responses import Response

text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16",
)
text2img_pipe = text2img_pipe.to("cuda")

img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
img2img_pipeline = img2img_pipeline.to("cuda")

app = FastAPI()


class Text2ImageInput(BaseModel):
    title: str
    content: str


@app.post("/text2img")
def text2img(input_payload: Text2ImageInput):
    with torch.inference_mode():
        image = text2img_pipe(input_payload.title).images[0]

    with io.BytesIO() as output_pointer:
        image.save(output_pointer, format="PNG")
        return Response(output_pointer.getvalue(), media_type="image/png")


@app.post("/img2img")
def img2img(title: str = Form(), content: str = Form(), file: UploadFile = File()):
    image_stream = io.BytesIO(file.file.read())
    image = Image.open(image_stream).convert("RGB")

    with torch.inference_mode():
        image = img2img_pipeline(prompt=title, init_image=image, strength=0.75, guidance_scale=7.5).images[0]

    with io.BytesIO() as output_pointer:
        image.save(output_pointer, format="PNG")
        return Response(output_pointer.getvalue(), media_type="image/png")
