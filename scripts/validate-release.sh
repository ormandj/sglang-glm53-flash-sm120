#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${1:-"$repo/release.json"}
lock=${2:-"$repo/stack.lock.json"}

for tool in jq grep; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done
jq -e . "$manifest" "$lock" >/dev/null

version=$(jq -er '.version' "$manifest")
candidate=$(jq -er '.candidate' "$manifest")
candidate_tag=$(jq -er '.candidate_tag' "$manifest")
stable_tag=$(jq -er '.stable_tag' "$manifest")
cache_schema=$(jq -er '.cache_schema' "$manifest")
product=$(jq -er '.product' "$manifest")

[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || {
  echo "release version is not strict stable SemVer: $version" >&2; exit 1;
}
[[ "$candidate" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "invalid candidate: $candidate" >&2; exit 1; }
[[ "$candidate_tag" == "v${version}-rc.${candidate}" ]] || { echo "invalid candidate tag" >&2; exit 1; }
[[ "$stable_tag" == "v${version}" ]] || { echo "invalid stable tag" >&2; exit 1; }
[[ "$cache_schema" =~ ^v[1-9][0-9]*$ ]] || { echo "invalid cache schema" >&2; exit 1; }
[[ "$product" == "sglang-glm53-flash-sm120" ]] || { echo "unexpected product: $product" >&2; exit 1; }

for field in version candidate candidate_tag stable_tag cache_schema; do
  [[ "$(jq -er ".${field}" "$manifest")" == "$(jq -er ".release.${field}" "$lock")" ]] || {
    echo "release.json and stack.lock.json disagree on $field" >&2; exit 1;
  }
done
[[ "$(jq -er '.image.tag' "$lock")" == "$candidate_tag" ]] || { echo "image tag mismatch" >&2; exit 1; }
[[ "$(jq -er '.image.platform' "$lock")" == "linux/amd64" ]] || { echo "unsupported platform" >&2; exit 1; }
[[ "$(jq -er '.model.native_context_length' "$lock")" == 1048576 ]] || { echo "native context mismatch" >&2; exit 1; }
[[ "$(jq -er '.hardware.compute_capability' "$lock")" == "sm_120" ]] || { echo "hardware scope mismatch" >&2; exit 1; }

check_pin() {
  local arg=$1 expected=$2 actual
  actual=$(jq -er --arg arg "$arg" '.pins[] | select(.arg == $arg) | .value' "$lock")
  [[ "$actual" == "$expected" ]] || { echo "$arg does not match release.json" >&2; exit 1; }
}
check_pin GLM53_RELEASE_VERSION "$version"
check_pin GLM53_RELEASE_CANDIDATE "$candidate"
check_pin GLM53_CACHE_SCHEMA "$cache_schema"
check_pin GLM53_MODEL_REVISION "$(jq -er '.model.source_revision' "$lock")"
check_pin GLM53_FLASHINFER_MAIN_HEAD "$(jq -er '.integration.flashinfer.head' "$lock")"
check_pin GLM53_FLASHINFER_MAIN_TREE "$(jq -er '.integration.flashinfer.tree' "$lock")"
check_pin GLM53_FLASHINFER_VERSION "$(jq -er '.integration.flashinfer.package_version' "$lock")"

launcher="$repo/examples/serve-glm53-flash.sh"
private_image="git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:${candidate_tag}"
grep -Fx "IMAGE=\${IMAGE:-${private_image}}" "$launcher" >/dev/null || {
  echo "launcher does not default to immutable private candidate" >&2; exit 1;
}
grep -F "/srv/cache/sglang-glm53-flash-sm120-${cache_schema}" "$repo/RUN.md" >/dev/null || {
  echo "RUN.md cache schema mismatch" >&2; exit 1;
}

echo "release contract valid: ${candidate_tag} -> ${stable_tag}, cache ${cache_schema}"
