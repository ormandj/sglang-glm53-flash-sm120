#!/usr/bin/env bash
# Target TP=2 profile for the local E4M3-K32 W4A16 artifact. The defaults are
# goals to qualify, not claims that v0.1.0-rc.45 has already achieved them.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the local GLM-5.3-Flash W4A16 artifact}"
: "${CACHE_DIR:?set CACHE_DIR to a candidate-specific persistent cache directory}"

IMAGE=${IMAGE:-sglang-glm53-flash-sm120:v0.1.0-rc.45}
PORT=${PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
TP_SIZE=${TP_SIZE:-2}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-524288}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-524288}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4}
# GLM's hybrid recurrent state consumes five physical slots per live request.
# C4 therefore needs 20 slots; this is model state, not ordinary KV cache.
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-20}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-4}
MEM_FRACTION=${MEM_FRACTION:-0.99}
CONTAINER_NAME=${CONTAINER_NAME:-glm53-flash-sm120}

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR does not contain config.json: $MODEL_DIR" >&2
  exit 2
fi
if [[ -e "$CACHE_DIR" && ! -d "$CACHE_DIR" ]]; then
  echo "CACHE_DIR exists but is not a directory: $CACHE_DIR" >&2
  exit 2
fi
if [[ "$TP_SIZE" != 2 ]]; then
  echo "v0.1.0-rc.45 is scoped to TP_SIZE=2" >&2
  exit 2
fi
for value in MAX_TOTAL_TOKENS MAX_RUNNING_REQUESTS MAX_MAMBA_CACHE_SIZE CUDA_GRAPH_MAX_BS; do
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

mkdir -p "$CACHE_DIR"
model_dir=$(cd "$MODEL_DIR" && pwd)
cache_dir=$(cd "$CACHE_DIR" && pwd)
container_model_path=/models/glm53-flash-w4a16-e4m3-k32

# EP is deliberately absent: TP=2 keeps half of the routed-expert bank on each
# GPU without adding low-concurrency all-to-all imbalance over PCIe. NCCL is
# used instead of SGLang custom all-reduce because this pair has no NVLink.
exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --entrypoint sglang \
  --gpus all \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --publish "${PORT}:8000" \
  --volume "${model_dir}:${container_model_path}:ro" \
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
  --served-model-name glm-5.3-flash \
  --tp "$TP_SIZE" \
  --enable-multimodal \
  --image-processor-backend pil \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --attention-backend dsa \
  --context-length "$CONTEXT_LENGTH" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "$MEM_FRACTION" \
  --chunked-prefill-size 8192 \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --mamba-ssm-dtype bfloat16 \
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS" \
  --dsa-prefill-backend flashinfer_sparse_mla \
  --dsa-decode-backend flashinfer_sparse_mla \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-metrics \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 8000
