ARG MODEL_IMAGE="model-dl"

FROM python:3.10 as env

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    libffi-dev \
    musl-dev \
    git \
    git-lfs && \
    git lfs install

ENV PYTHONIOENCODING=utf-8
ENV MKL_NUM_THREADS=""

WORKDIR /app

RUN adduser --system --group app && chown -R app:app /app
USER app

ENV PATH="/home/app/.local/bin:${PATH}"

COPY --chown=app:app requirements.txt .
RUN pip install --user -r requirements.txt && \
    rm requirements.txt && \
    python -c "import nltk; nltk.download(\"punkt\")"

ENTRYPOINT ["python", "main.py"]

FROM alpine as model-dl

RUN apk update && \
    apk add git git-lfs yq && \
    git lfs install

ARG MODEL_CONFIG_FILE
COPY ${MODEL_CONFIG_FILE} /models/config.yaml

RUN HF_MODEL=$(yq '.huggingface' /models/config.yaml) &&  \
    MODEL_ROOT=$(yq '.model_root' /models/config.yaml) &&  \
    git lfs clone --progress https://huggingface.co/$HF_MODEL $MODEL_ROOT

FROM $MODEL_IMAGE as model

FROM env as worker-model

COPY --chown=app:app --from=model /models /app/models
COPY --chown=app:app . .

FROM env as worker-base

COPY --chown=app:app . .