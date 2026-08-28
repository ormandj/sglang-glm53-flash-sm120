#!/usr/bin/env bash
# Candidate serving envelope for GLM-5.3-Flash MXFP4 on two RTX PRO 6000
# Blackwell GPUs (SM120, TP=2). It is not a qualification claim until the
# primary repository's RESULTS.md records a passing run.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the GLM-5.3-Flash-BF16-MXFP4 artifact directory}"
: "${CACHE_DIR:?set CACHE_DIR to a persistent, image-specific cache directory}"

IMAGE=${IMAGE:-git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.5}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
TP_SIZE=${TP_SIZE:-2}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-524288}
MEM_FRACTION=${MEM_FRACTION:-0.96}
# Recurrent state is allocated per THIS value, not per live request:
# 72.78 MiB/slot for the 34 KDA layers. 32 slots would silently reserve
# 2.3 GiB of the KV pool at BF16, or 4.5 GiB if the SSM dtype were left FP32.
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}
# mtp preserves the checkpoint's native layer-45 NEXTN block. dflash uses the
# separately pinned DFlash2 assistant. none is the verifier-only A/B baseline.
SPECULATIVE_MODE=${SPECULATIVE_MODE:-mtp}
DFLASH_DIR=${DFLASH_DIR:-}
CONTAINER_NAME=${CONTAINER_NAME:-glm53-flash-sm120}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR does not contain config.json: $MODEL_DIR" >&2
  exit 2
fi
if [[ -e "$CACHE_DIR" && ! -d "$CACHE_DIR" ]]; then
  echo "CACHE_DIR exists but is not a directory: $CACHE_DIR" >&2
  exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  echo "PORT must be an integer from 1 through 65535" >&2
  exit 2
fi
if [[ "$TP_SIZE" != 2 ]]; then
  echo "v0.1.0-rc.5 is scoped to TP_SIZE=2" >&2
  exit 2
fi
if [[ "$SPECULATIVE_MODE" != none && "$SPECULATIVE_MODE" != mtp && "$SPECULATIVE_MODE" != dflash ]]; then
  echo "SPECULATIVE_MODE must be none, mtp, or dflash" >&2
  exit 2
fi
if ! [[ "$MAX_RUNNING_REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNNING_REQUESTS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
model_dir=$(cd "$MODEL_DIR" && pwd)
cache_dir=$(cd "$CACHE_DIR" && pwd)
container_model_path=/models/zai-org/GLM-5.3-Flash-BF16-MXFP4
container_dflash_path=/models/incoai/GLM-5.3-Flash-DFlash2

speculative_args=()
mount_args=(--volume "${model_dir}:${container_model_path}:ro")
case "$SPECULATIVE_MODE" in
  none)
    ;;
  mtp)
    speculative_args+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 5
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 6
      --speculative-adaptive
      --speculative-draft-model-quantization compressed-tensors
      --speculative-moe-runner-backend flashinfer_mxfp4
    )
    ;;
  dflash)
    if [[ -z "$DFLASH_DIR" || ! -f "$DFLASH_DIR/config.json" ]]; then
      echo "DFLASH_DIR must contain the pinned GLM-5.3-Flash-DFlash2 config.json" >&2
      exit 2
    fi
    dflash_dir=$(cd "$DFLASH_DIR" && pwd)
    mount_args+=(--volume "${dflash_dir}:${container_dflash_path}:ro")
    speculative_args+=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "$container_dflash_path"
      --speculative-dflash-block-size 8
      --speculative-draft-window-size 2048
      --speculative-draft-kv-cache-dtype fp8_e4m3
      --speculative-draft-attention-backend fa4
      --speculative-draft-model-quantization unquant
    )
    ;;
esac

# Expert Parallel is deliberately NOT used. At TP=2 it saves essentially no
# VRAM (both layouts store ~half the expert bytes per GPU) while introducing
# ~27% expected routing imbalance at C=1 and an all-to-all over PCIe.
#
# HiCache is deliberately absent from v0.1. It is a prefix-reuse tier, not a
# max-context lever, and it needs a bounded dedicated volume rather than an
# implicit host-path side effect.
exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --entrypoint sglang \
  --gpus all \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --publish "${PORT}:8000" \
  "${mount_args[@]}" \
  --volume "${cache_dir}:/root/.cache" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  "$IMAGE" \
  serve \
  --model-path "$container_model_path" \
  --served-model-name glm53-flash-sm120 \
  --tp "$TP_SIZE" \
  --enable-multimodal \
  --quantization compressed-tensors \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-shared-experts-fusion \
  --context-length "$CONTEXT_LENGTH" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "$MEM_FRACTION" \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-decode 8 \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --mamba-ssm-dtype bfloat16 \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-metrics \
  --enable-cache-report \
  "${speculative_args[@]+"${speculative_args[@]}"}" \
  --host 0.0.0.0 \
  --port 8000
