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

for file in README.md RUN.md CHANGELOG.md AGENTS.md NOTICE.md "$launcher"; do
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

if grep -E -- '^## Releases$|git\.home\.corenode\.com|current internal stable image|^\| Qualified internal candidate \|' "$repo/README.md" >/dev/null; then
  echo "README must not contain internal registry bookkeeping or a release-history section; keep operational records in the private project" >&2
  exit 1
fi

for file in README.md CHANGELOG.md AGENTS.md; do
  if grep -E -- 'git\.home\.corenode\.com|/Users/ormandj/|~/git/homelab/' "$repo/$file" >/dev/null; then
    echo "$file contains private operational details" >&2
    exit 1
  fi
done

critical=(
  'TP_SIZE=${TP_SIZE:-2}'
  'CONTEXT_LENGTH=${CONTEXT_LENGTH:-450560}'
  'MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-450560}'
  'MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4}'
  'MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-28}'
  'CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-4}'
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

if grep -E -- '(^|[[:space:]])--ep([[:space:]]|$)|EP_SIZE' "$launcher" >/dev/null; then
  echo "launcher must not enable Expert Parallel at TP=2" >&2
  exit 1
fi
if grep -F -- 'flashinfer_mxfp4' "$launcher" >/dev/null; then
  echo "launcher carries the rejected MXFP4 runner" >&2
  exit 1
fi
if grep -F -- '--trust-remote-code' "$launcher" >/dev/null; then
  echo "native GLM support must not require trust-remote-code" >&2
  exit 1
fi

echo "documentation contract valid: ${candidate_tag}, cache ${cache_schema}, TP2 vision+MTP profile"
