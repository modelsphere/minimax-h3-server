#!/usr/bin/env bash
set -e

data_dir="${H3_DATA_DIR:-/srv/minimax-h3}"
model_dir="${H3_FL2VA_MODEL_DIR:-/srv/models/MiniMax-H3/FL2VA}"
mkdir -p "$data_dir/sglang-output"
docker rm -f minimax-h3-sglang 2>/dev/null || true
docker run -d \
  --name minimax-h3-sglang \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 30010:30010 \
  -v "$model_dir:/model:ro" \
  -v "$data_dir:/data/minimax-h3" \
  "${H3_SGLANG_IMAGE:-lmsysorg/sglang:v0.5.19}" \
  sglang serve \
    --model-type diffusion \
    --model-path /model \
    --model-subfolder . \
    --pipeline-class-name MiniMaxH3Pipeline \
    --trust-remote-code true \
    --num-gpus 4 \
    --tp-size 2 \
    --ulysses-degree 2 \
    --encoder-parallel auto \
    --performance-mode speed \
    --enable-torch-compile false \
    --output-path /data/minimax-h3/sglang-output \
    --host 0.0.0.0 \
    --port 30010
