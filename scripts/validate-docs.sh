#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
candidate_tag=$(jq -er '.candidate_tag' "$repo/release.json")
cache_schema=$(jq -er '.cache_schema' "$repo/release.json")
local_image="sglang-glm53-flash-sm120:${candidate_tag}"
launcher="$repo/examples/serve-glm53-flash.sh"

for file in README.md RUN.md CHANGELOG.md AGENTS.md NOTICE.md "$launcher"; do
  [[ -s "$repo/$file" || -s "$file" ]] || { echo "required file missing: $file" >&2; exit 1; }
done

require_text() {
  local file=$1 expected=$2
  grep -F -- "$expected" "$file" >/dev/null || { echo "${file#$repo/} missing: $expected" >&2; exit 1; }
}

require_text "$repo/README.md" "$local_image"
require_text "$repo/RUN.md" "IMAGE=${local_image}"
require_text "$launcher" "IMAGE=\${IMAGE:-${local_image}}"
require_text "$repo/RUN.md" "/srv/cache/sglang-glm53-flash-sm120-${cache_schema}"
require_text "$repo/CHANGELOG.md" "## ${candidate_tag}"
require_text "$repo/AGENTS.md" 'always uses the complete release name'

critical=(
  'TP_SIZE=${TP_SIZE:-2}'
  'CONTEXT_LENGTH=${CONTEXT_LENGTH:-524288}'
  'MEM_FRACTION=${MEM_FRACTION:-0.96}'
  'MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}'
  'SPECULATIVE_MODE=${SPECULATIVE_MODE:-mtp}'
  '--enable-multimodal'
  '--quantization compressed-tensors'
  '--kv-cache-dtype fp8_e4m3'
  '--chunked-prefill-size 8192'
  '--cuda-graph-max-bs-decode 8'
  '--mamba-ssm-dtype bfloat16'
  '--moe-runner-backend flashinfer_mxfp4'
  '--speculative-algorithm EAGLE'
  '--speculative-num-steps 5'
  '--speculative-num-draft-tokens 6'
  '--speculative-adaptive'
  '--speculative-draft-model-quantization compressed-tensors'
  '--speculative-moe-runner-backend flashinfer_mxfp4'
  '--speculative-algorithm DFLASH'
  '--speculative-dflash-block-size 8'
  '--speculative-draft-window-size 2048'
  '--speculative-draft-kv-cache-dtype fp8_e4m3'
  '--speculative-draft-attention-backend fa4'
  '--speculative-draft-model-quantization unquant'
  '--dsa-prefill-backend flashinfer_sparse_mla'
  '--dsa-decode-backend flashinfer_sparse_mla'
  '--reasoning-parser glm45'
  '--tool-call-parser glm47'
)
for value in "${critical[@]}"; do require_text "$launcher" "$value"; done

if grep -E -- '--enable-hierarchical-cache|--hicache-|SGLANG_HICACHE' "$launcher" >/dev/null; then
  echo "v0.1 launcher unexpectedly enables HiCache" >&2
  exit 1
fi
if grep -F -- '--trust-remote-code' "$launcher" >/dev/null; then
  echo "glm5_next is native to the base image; trust-remote-code must not be needed" >&2
  exit 1
fi
if grep -F -- '--disable-custom-all-reduce' "$launcher" >/dev/null; then
  echo "launcher must allow SGLang to test two-GPU PCIe custom all-reduce" >&2
  exit 1
fi
# Expert Parallel is a deliberate non-choice at TP=2: ~0 VRAM saved, ~27%
# routing imbalance at C=1. Catch it being reintroduced silently.
if grep -E -- '(^|[[:space:]])--ep([[:space:]]|$)|EP_SIZE' "$launcher" >/dev/null; then
  echo "launcher must not enable Expert Parallel at TP=2" >&2
  exit 1
fi
# --mamba-ssm-dtype is mandatory: SGLang defaults the SSM state to FP32, which
# costs 2.2 GiB of KV pool at max-running-requests=8.
grep -F -- '--mamba-ssm-dtype bfloat16' "$launcher" >/dev/null || {
  echo "launcher must pin --mamba-ssm-dtype bfloat16" >&2; exit 1; }

echo "documentation contract valid: ${candidate_tag}, cache ${cache_schema}, SM120-only candidate"
