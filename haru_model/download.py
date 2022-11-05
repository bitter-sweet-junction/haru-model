import argparse

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import HfFolder

parser = argparse.ArgumentParser()
parser.add_argument("--hf-token", type=str, required=True)


def main(args: argparse.Namespace):
    HfFolder.save_token(args.hf_token)
    StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        revision="fp16",
    )
    print("download complete")


if __name__ == "__main__":
    main(parser.parse_args())
