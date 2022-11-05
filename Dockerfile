FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt --no-cache

COPY ./haru_model ./haru_model

# download model from huggingface model hub and save to docker image
ARG HF_TOKEN
RUN python -m haru_model.download --hf-token $HF_TOKEN

ENTRYPOINT ["uvicorn", "haru_model.app:app"]
CMD ["--host", "0.0.0.0", "--port", "8888"]


