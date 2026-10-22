# RETICLE runtime: expert training and inference, the evidence library, the agent client, evaluation and the demo server.
# The vision-language model (Qwen3.8-27B) is served separately with vLLM (see README.md).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# MUSK model code (weights are downloaded from the Hugging Face hub, xiangjx/musk)
RUN git clone https://github.com/lilab-stanford/MUSK.git /opt/MUSK && git -C /opt/MUSK checkout 714b666969c1911e5efe70d991140a21030f4ef3

ENV MUSK_SRC=/opt/MUSK PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
WORKDIR /workspace
