#!/usr/bin/env bash
set -e

# Ref2VA SGLang。H3 API 和本容器必须能在 /data/minimax-h3 看到同一份输入素材。
# 首尾帧与参考素材混用需要 SGLang v0.5.19 或更新版本。
# 两个启动脚本使用相同的 H3_REF2VA_TURBO_ENABLED(默认 true)。
source "$(dirname "${BASH_SOURCE[0]}")/ref2va_profile.sh"
ref2va_configure
if [[ "$ref2va_turbo_enabled" == true && ! -f "$ref2va_lora_dir/$ref2va_lora_file" ]]; then
  echo "Missing Ref2VA Turbo LoRA: $ref2va_lora_dir/$ref2va_lora_file" >&2
  exit 1
fi
data_dir="${H3_DATA_DIR:-/srv/minimax-h3}"
output_dir="${H3_REF2VA_OUTPUT_DIR:-$data_dir/ref2va-output}"
model_dir="${H3_REF2VA_MODEL_DIR:-/srv/models/MiniMax-H3/Ref2VA}"
mkdir -p "$output_dir" "$data_dir"
docker rm -f minimax-h3-sglang-ref2va 2>/dev/null || true
docker run -d \
  --name minimax-h3-sglang-ref2va \
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
  "${ref2va_lora_mount_args[@]}" \
  -v "$data_dir:/data/minimax-h3:ro" \
  -v "$output_dir:/output" \
  "${H3_SGLANG_IMAGE:-lmsysorg/sglang:v0.5.19}" \
  sglang serve \
    --model-type diffusion \
    --model-path /model \
    --model-subfolder . \
    --pipeline-class-name MiniMaxH3Pipeline \
    --trust-remote-code true \
    --num-gpus 4 \
    --tp-size 4 \
    --ulysses-degree 1 \
    --encoder-parallel auto \
    --performance-mode speed \
    --enable-torch-compile false \
    --warmup-mode request \
    --warmup-steps 2 \
    "${ref2va_lora_args[@]}" \
    --output-path /output \
    --host 0.0.0.0 \
    --port 30010
