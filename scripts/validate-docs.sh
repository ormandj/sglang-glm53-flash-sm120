#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
candidate_tag=$(jq -er '.candidate_tag' "$repo/release.json")
cache_schema=$(jq -er '.cache_schema' "$repo/release.json")
stable_tag=$(jq -er '.stable_tag' "$repo/release.json")
local_image="sglang-glm53-flash-sm120:${candidate_tag}"
# Once the current release is public, run instructions and the launcher must
# select its immutable stable image. Candidate docs keep the candidate default.
if grep -F -- "The current published stable image is \`${stable_tag}\`" "$repo/README.md" >/dev/null; then
  local_image="ghcr.io/ormandj/sglang-glm53-flash-sm120:${stable_tag}"
fi
launcher="$repo/examples/serve-glm53-flash.sh"
launcher_tp4="$repo/examples/serve-glm53-flash-tp4.sh"

for file in README.md RUN.md CHANGELOG.md AGENTS.md NOTICE.md BENCHMARKS.md "$launcher" "$launcher_tp4"; do
  [[ -s "$repo/$file" || -s "$file" ]] || { echo "required file missing: $file" >&2; exit 1; }
done

require_text() {
  local file=$1 expected=$2
  grep -F -- "$expected" "$file" >/dev/null || {
    echo "${file#$repo/} missing: $expected" >&2; exit 1;
  }
}

require_text "$repo/README.md" "$local_image"
require_text "$repo/RUN.md" "IMAGE=${local_image}"
require_text "$launcher" "IMAGE=\${IMAGE:-${local_image}}"
require_text "$repo/RUN.md" "/srv/cache/sglang-glm53-flash-sm120-${cache_schema}"
require_text "$repo/CHANGELOG.md" "# Changelog"
require_text "$repo/AGENTS.md" 'always uses the complete release name'

if grep -E -- '^## Releases$|([a-z0-9-]+\.)?home\.[a-z0-9.-]+|registry\.internal\.example|current internal stable image|^\| Qualified internal candidate \|' "$repo/README.md" >/dev/null; then
  echo "README must not contain internal registry bookkeeping or a release-history section; keep operational records in the private project" >&2
  exit 1
fi

for file in README.md RUN.md BENCHMARKS.md CHANGELOG.md AGENTS.md NOTICE.md; do
  if grep -E -- '([a-z0-9-]+\.)?home\.[a-z0-9.-]+|registry\.internal\.example|/Users/[^/]+/|~/git/[^/]+/|(^|[^[:alnum:]_])(this|our) homelab([^[:alnum:]_]|$)' "$repo/$file" >/dev/null; then
    echo "$file contains private operational details" >&2
    exit 1
  fi
done

critical=(
  'TP_SIZE=${TP_SIZE:-2}'
  'CONTEXT_LENGTH=${CONTEXT_LENGTH:-524288}'
  'MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-524288}'
  'MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4}'
  'MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-28}'
  'CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-4}'
  '--cuda-graph-backend-prefill breakable'
  '--cuda-graph-bs-prefill 64 128'
  '--env SGLANG_BCG_RAGGED_SHAPES=1'
  '--env SGLANG_BCG_RAGGED_MAX_BS=1'
  '--env SGLANG_BCG_SEPARATE_CAPTURE_SESSIONS=1'
  '--enable-multimodal'
  '--image-processor-backend torchvision'
  '--mm-preprocessing-device cpu'
  '--env SGLANG_KDA_EXTEND_BLOCK_TOKENS=2048'
  '--moe-runner-backend flashinfer_cutlass'
  '--kv-cache-dtype fp8_e4m3'
  '--dsa-prefill-backend flashinfer_sparse_mla'
  '--dsa-decode-backend flashinfer_sparse_mla'
  '--mamba-ssm-dtype bfloat16'
  '--speculative-algorithm EAGLE'
  '--speculative-num-steps 5'
  '--speculative-eagle-topk 1'
  '--speculative-num-draft-tokens 6'
  '--speculative-adaptive'
  '--reasoning-parser glm45'
  '--tool-call-parser glm47'
)
for value in "${critical[@]}"; do require_text "$launcher" "$value"; done

tp4_critical=(
  "IMAGE=\${IMAGE:-${local_image}}"
  'TP_SIZE=${TP_SIZE:-4}'
  'CONTEXT_LENGTH=${CONTEXT_LENGTH:-1048576}'
  'MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-2621440}'
  'MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}'
  'MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-224}'
  'HICACHE_SIZE_GB=${HICACHE_SIZE_GB:-40}'
  '--quantization modelopt_fp4'
  '--chunked-prefill-size 8192 --max-prefill-tokens 8192'
  '--cuda-graph-bs-decode 1 2 3 4 8 16 32'
  '--speculative-draft-model-quantization unquant'
  '--json-model-override-args'
  'model.layers.45.*'
  '--env NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"'
)
for value in "${tp4_critical[@]}"; do require_text "$launcher_tp4" "$value"; done
for value in '--enable-multimodal' '--mm-preprocessing-device cpu' '--image-processor-backend torchvision' '--kv-cache-dtype fp8_e4m3' '--dsa-prefill-backend flashinfer_sparse_mla' '--dsa-decode-backend flashinfer_sparse_mla' '--speculative-adaptive' '--reasoning-parser glm45' '--tool-call-parser glm47'; do
  require_text "$launcher_tp4" "$value"
done
require_text "$launcher" '--quantization modelopt_mixed'
require_text "$repo/README.md" 'ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO'
require_text "$repo/README.md" 'nvidia/GLM-5.3-Flash-NVFP4'
require_text "$repo/BENCHMARKS.md" "TP2 performance was not measured for ${stable_tag}"

for profile in "$launcher" "$launcher_tp4"; do
  if grep -E -- '(^|[[:space:]])--ep([[:space:]]|$)|EP_SIZE' "$profile" >/dev/null; then
    echo "launcher must keep expert parallel 1: $profile" >&2
    exit 1
  fi
  if grep -E -- 'flashinfer_mxfp4|--trust-remote-code' "$profile" >/dev/null; then
    echo "launcher carries an unsupported runner or remote-code setting: $profile" >&2
    exit 1
  fi
done

echo "documentation contract valid: ${stable_tag}, cache ${cache_schema}, TP2 W4A16 and TP4 NVIDIA NVFP4 vision+MTP profiles"
