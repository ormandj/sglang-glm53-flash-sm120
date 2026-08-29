#!/usr/bin/env bash
# Reproduce every source tree used by this candidate and verify the immutable
# base manifest. The historical vendor-byte patches are intentionally absent.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
lock=stack.lock.json

for tool in jq git curl; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }
done
"$repo/scripts/validate-release.sh"

if [[ "$(jq -er '.patches | length' "$lock")" != 0 ]]; then
  echo "v0.1.0-rc.34 must not carry archived runtime patches" >&2
  exit 1
fi
if [[ -d patches && -n "$(find patches -type f -print -quit)" ]]; then
  echo "patches/ contains unrecorded files" >&2
  exit 1
fi

pin() { jq -er --arg arg "$1" '.pins[] | select(.arg == $arg) | .value' "$lock"; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

verify_tree() {
  local name=$1 repository=$2 head=$3 tree=$4 destination="$work/$1"
  git init -q "$destination"
  git -C "$destination" remote add origin "$repository"
  git -C "$destination" fetch -q --depth=1 origin "$head"
  git -C "$destination" checkout -q --detach FETCH_HEAD
  local actual_head actual_tree
  actual_head=$(git -C "$destination" rev-parse HEAD)
  actual_tree=$(git -C "$destination" rev-parse 'HEAD^{tree}')
  [[ "$actual_head" == "$head" ]] || { echo "$name head mismatch" >&2; exit 1; }
  [[ "$actual_tree" == "$tree" ]] || { echo "$name tree mismatch" >&2; exit 1; }
  printf '  %s head %s tree %s\n' "$name" "$actual_head" "$actual_tree"
}

echo "== source trees =="
verify_tree sglang \
  "$(jq -er '.integration.sglang.repository' "$lock")" \
  "$(pin GLM53_SGLANG_HEAD)" "$(pin GLM53_SGLANG_TREE)"
verify_tree flashinfer \
  "$(jq -er '.integration.flashinfer.repository' "$lock")" \
  "$(pin GLM53_FLASHINFER_HEAD)" "$(pin GLM53_FLASHINFER_TREE)"
verify_tree modelopt \
  "$(jq -er '.integration.modelopt.repository' "$lock")" \
  "$(pin GLM53_MODELOPT_HEAD)" "$(pin GLM53_MODELOPT_TREE)"
git -C "$work/modelopt" fetch -q --depth=1 origin \
  "refs/tags/$(pin GLM53_MODELOPT_RELEASE_TAG):refs/tags/$(pin GLM53_MODELOPT_RELEASE_TAG)"
[[ "$(git -C "$work/modelopt" rev-parse "$(pin GLM53_MODELOPT_RELEASE_TAG)^{commit}")" == \
   "$(pin GLM53_MODELOPT_HEAD)" ]] || { echo "modelopt release tag mismatch" >&2; exit 1; }
printf '  modelopt release tag %s\n' "$(pin GLM53_MODELOPT_RELEASE_TAG)"

[[ "$(jq -er '.verification.sglang_source_verifiable' "$lock")" == true ]]
[[ "$(jq -er '.verification.sglang_repository' "$lock")" == "$(pin GLM53_SGLANG_REPOSITORY)" ]]

echo "== base image manifests =="
token=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:lmsysorg/sglang:pull" | jq -er .token)
accept='application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json'
for digest in "$(pin GLM53_SGLANG_BASE_INDEX)" "$(pin GLM53_SGLANG_BASE_AMD64_MANIFEST)"; do
  curl -fsSL -o /dev/null \
    -H "Authorization: Bearer ${token}" -H "Accept: ${accept}" \
    "https://registry-1.docker.io/v2/lmsysorg/sglang/manifests/${digest}"
  printf '  %s\n' "$digest"
done

echo "release bundle verified: exact SGLang, FlashInfer, and ModelOpt trees; no local patches"
