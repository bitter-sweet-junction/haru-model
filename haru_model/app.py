import io
from datetime import datetime
from typing import Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel
from starlette.responses import FileResponse

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16",
    scheduler=scheduler,
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
    content: Optional[str] = None


@app.post("/text2img")
def text2img(input_payload: Text2ImageInput):
    # content = input_payload.content if input_payload.content else ""
    # title = response["choices"][0]["text"].strip()

    # response = openai.Completion.create(
    #     model="text-ada-001",
    #     prompt=f"Abstract diary into a image title less 8 words.\n{input_payload.title}. {content}\nThe image of ",
    #     temperature=0.7,
    #     max_tokens=32,
    #     top_p=0.9,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    # )

    title = f"A Cosy image of {input_payload.title}. highly detailed, art by Makoto Shinkai, Trending on artstation"
    print(title)
    with torch.inference_mode():
        image = text2img_pipe(title, num_inference_steps=100).images[0]

    datestring = datetime.now().isoformat()
    file_name = f"logs/{datestring}.png"
    image.save(file_name, format="PNG")

    with open(f"logs/{datestring}.txt", "w") as f:
        f.write(title + "\n")

    return FileResponse(file_name)


@app.post("/img2img")
def img2img(title: str = Form(), content: str = Form(), file: UploadFile = File()):
    image_stream = io.BytesIO(file.file.read())
    image = Image.open(image_stream).convert("RGB")

    with torch.inference_mode():
        image = img2img_pipeline(
            prompt=title, init_image=image, strength=0.75, guidance_scale=7.5, num_inference_steps=200
        ).images[0]

    datestring = datetime.now().isoformat()
    file_name = f"logs/{datestring}.png"
    image.save(file_name, format="PNG")

    return FileResponse(file_name)
