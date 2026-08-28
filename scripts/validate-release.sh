#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${1:-"$repo/release.json"}
lock=${2:-"$repo/stack.lock.json"}

for tool in jq grep sed; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
jq -e . "$manifest" "$lock" >/dev/null

version=$(jq -er '.version' "$manifest")
candidate=$(jq -er '.candidate' "$manifest")
candidate_tag=$(jq -er '.candidate_tag' "$manifest")
stable_tag=$(jq -er '.stable_tag' "$manifest")
cache_schema=$(jq -er '.cache_schema' "$manifest")
product=$(jq -er '.product' "$manifest")

[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]
[[ "$candidate" =~ ^(0|[1-9][0-9]*)$ ]]
[[ "$candidate_tag" == "v${version}-rc.${candidate}" ]]
[[ "$stable_tag" == "v${version}" ]]
[[ "$cache_schema" =~ ^v[1-9][0-9]*$ ]]
[[ "$product" == "sglang-glm53-flash-sm120" ]]

for field in version candidate candidate_tag stable_tag cache_schema; do
  [[ "$(jq -er ".${field}" "$manifest")" == "$(jq -er ".release.${field}" "$lock")" ]] || {
    echo "release.json and stack.lock.json disagree on $field" >&2; exit 1;
  }
done
[[ "$(jq -er '.image.tag' "$lock")" == "$candidate_tag" ]]
[[ "$(jq -er '.image.platform' "$lock")" == "linux/amd64" ]]
[[ "$(jq -er '.model.native_context_length' "$lock")" == 1048576 ]]
[[ "$(jq -er '.hardware.compute_capability' "$lock")" == "sm_120" ]]
[[ "$(jq -er '.hardware.tensor_parallel' "$lock")" == 2 ]]
[[ "$(jq -er '.hardware.expert_parallel' "$lock")" == 1 ]]

check_pin() {
  local arg=$1 expected=$2 actual count
  count=$(jq -er --arg arg "$arg" '[.pins[] | select(.arg == $arg)] | length' "$lock")
  [[ "$count" == 1 ]] || { echo "expected exactly one pin for $arg" >&2; exit 1; }
  actual=$(jq -er --arg arg "$arg" '.pins[] | select(.arg == $arg) | .value' "$lock")
  [[ "$actual" == "$expected" ]] || { echo "$arg mismatch" >&2; exit 1; }
}
check_pin GLM53_RELEASE_VERSION "$version"
check_pin GLM53_RELEASE_CANDIDATE "$candidate"
check_pin GLM53_CACHE_SCHEMA "$cache_schema"
check_pin GLM53_MODEL_REVISION "$(jq -er '.model.source_revision' "$lock")"
check_pin GLM53_SGLANG_HEAD "$(jq -er '.integration.sglang.head' "$lock")"
check_pin GLM53_SGLANG_TREE "$(jq -er '.integration.sglang.tree' "$lock")"
check_pin GLM53_FLASHINFER_HEAD "$(jq -er '.integration.flashinfer.head' "$lock")"
check_pin GLM53_FLASHINFER_TREE "$(jq -er '.integration.flashinfer.tree' "$lock")"
check_pin GLM53_FLASHINFER_VERSION "$(jq -er '.integration.flashinfer.package_version' "$lock")"
check_pin GLM53_MODELOPT_HEAD "$(jq -er '.integration.modelopt.head' "$lock")"
check_pin GLM53_MODELOPT_TREE "$(jq -er '.integration.modelopt.tree' "$lock")"
check_pin GLM53_MODELOPT_VERSION "$(jq -er '.integration.modelopt.package_version' "$lock")"

while IFS=$'\t' read -r arg value; do
  grep -Fxq "ARG ${arg}=${value}" "$repo/Containerfile" || {
    echo "Containerfile pin mismatch: $arg" >&2; exit 1;
  }
done < <(jq -r '.pins[] | [.arg, .value] | @tsv' "$lock")
while IFS= read -r arg; do
  [[ -n "$arg" ]] || continue
  [[ "$(jq -r --arg arg "$arg" '[.pins[] | select(.arg == $arg)] | length' "$lock")" == 1 ]] || {
    echo "unrecorded Containerfile pin: $arg" >&2; exit 1;
  }
done < <(sed -n 's/^ARG \([A-Z0-9_]*\)=.*/\1/p' "$repo/Containerfile")

launcher="$repo/examples/serve-glm53-flash.sh"
local_image="sglang-glm53-flash-sm120:${candidate_tag}"
grep -Fx "IMAGE=\${IMAGE:-${local_image}}" "$launcher" >/dev/null
grep -F "/srv/cache/sglang-glm53-flash-sm120-${cache_schema}" "$repo/RUN.md" >/dev/null

echo "release contract valid: ${candidate_tag} -> ${stable_tag}, cache ${cache_schema}"
