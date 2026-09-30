#!/usr/bin/env bash
# TP4 profile for nvidia/GLM-5.3-Flash-NVFP4 on four 96 GB SM120 GPUs.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the local NVIDIA GLM-5.3-Flash-NVFP4 artifact}"
: "${CACHE_DIR:?set CACHE_DIR to a version-specific persistent cache directory}"

IMAGE=${IMAGE:-ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
TP_SIZE=${TP_SIZE:-4}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-1048576}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-2621440}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
# BF16 recurrent-state slots are separate from the shared FP8 KV pool.
# Keep enough slots for every admitted request and speculative intermediates.
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-224}

# HiCache uses 40 decimal GB per rank (160 GB across TP4), in addition to
# loading and serving memory. Set ENABLE_HICACHE=0 to disable the host tier.
ENABLE_HICACHE=${ENABLE_HICACHE:-1}
HICACHE_SIZE_GB=${HICACHE_SIZE_GB:-40}
HICACHE_ARGS=()
if [[ "$ENABLE_HICACHE" == 1 ]]; then
  HICACHE_ARGS=(--enable-hierarchical-cache --hicache-size "$HICACHE_SIZE_GB")
fi
MEM_FRACTION=${MEM_FRACTION:-0.99}
CONTAINER_NAME=${CONTAINER_NAME:-glm53-flash-sm120-tp4}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR does not contain config.json: $MODEL_DIR" >&2
  exit 2
fi
if [[ -e "$CACHE_DIR" && ! -d "$CACHE_DIR" ]]; then
  echo "CACHE_DIR exists but is not a directory: $CACHE_DIR" >&2
  exit 2
fi
if [[ "$TP_SIZE" != 4 ]]; then
  echo "This launcher is scoped to TP_SIZE=4" >&2
  exit 2
fi
for value in MAX_TOTAL_TOKENS MAX_RUNNING_REQUESTS MAX_MAMBA_CACHE_SIZE CONTEXT_LENGTH; do
  current=${!value}
  if ! [[ "$current" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value must be a positive integer" >&2
    exit 2
  fi
done
if (( MAX_MAMBA_CACHE_SIZE < MAX_RUNNING_REQUESTS * 5 )); then
  echo "MAX_MAMBA_CACHE_SIZE must be at least 5 * MAX_RUNNING_REQUESTS" >&2
  exit 2
fi

if (( MAX_RUNNING_REQUESTS > 32 )); then
  echo "This TP4 profile supports at most 32 running requests" >&2
  exit 2
fi
if (( CONTEXT_LENGTH > 1048576 )); then
  echo "CONTEXT_LENGTH exceeds the checkpoint's 1048576-token limit" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
model_dir=$(cd "$MODEL_DIR" && pwd)
cache_dir=$(cd "$CACHE_DIR" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The adaptive spec-decode step config is REQUIRED at long context: without
# it the adaptive controller sizes its speculative state buffers for the
# default step ladder, which scales with the token pool and OOMs once real
# context accumulates. The shipped config pins candidate_steps [3, 5].
ADAPTIVE_CONFIG=${ADAPTIVE_CONFIG:-$script_dir/glm53-adaptive.json}
if [[ ! -f "$ADAPTIVE_CONFIG" ]]; then
  echo "ADAPTIVE_CONFIG not found: $ADAPTIVE_CONFIG" >&2
  exit 2
fi
container_model_path=/models/glm53-flash-nvfp4

# TP4/EP1 matches the four-GPU PCIe profile. SYS permits NCCL peer access
# across CPU roots; the driver and platform must provide working GPU P2P.
exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --entrypoint sglang \
  --gpus all \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --publish "${PORT}:8000" \
  --volume "${model_dir}:${container_model_path}:ro" \
  --volume "${cache_dir}:/root/.cache" \
  --volume "${ADAPTIVE_CONFIG}:/etc/glm53-adaptive/adaptive.json:ro" \
  --env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8 \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
  --env SGLANG_PCIE_IPC_MAX_NUMEL=786432 \
  --env SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1 \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  --env SGLANG_KDA_EXTEND_BLOCK_TOKENS=2048 \
  --env SGLANG_BCG_RAGGED_SHAPES=1 \
  --env SGLANG_BCG_RAGGED_MAX_BS=1 \
  --env SGLANG_BCG_SEPARATE_CAPTURE_SESSIONS=1 \
  "$IMAGE" \
  serve \
  --model-path "$container_model_path" \
  --served-model-name glm-5.3-flash \
  --tp "$TP_SIZE" \
  --enable-multimodal \
  --warmups serving_coverage \
  --image-processor-backend torchvision \
  --mm-preprocessing-device cpu \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --attention-backend dsa \
  --linear-attn-backend triton \
  --context-length "$CONTEXT_LENGTH" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "$MEM_FRACTION" \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --mamba-ssm-dtype bfloat16 \
  "${HICACHE_ARGS[@]}" \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 64 128 \
  --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 1 2 3 4 8 16 32 \
  --dsa-prefill-backend flashinfer_sparse_mla \
  --dsa-decode-backend flashinfer_sparse_mla \
  --json-model-override-args '{"text_config":{"quantization_config":{"ignore":["model.layers.45.*"]}}}' \
  --speculative-draft-model-quantization unquant \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --speculative-adaptive-config /etc/glm53-adaptive/adaptive.json \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-metrics \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 8000
