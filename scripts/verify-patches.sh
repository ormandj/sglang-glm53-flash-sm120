#!/usr/bin/env bash
# Reproduce every source tree used by this candidate and verify the immutable
# base manifest. Project-owned deltas are checksummed patches over exact
# official-upstream commits; no cross-repository build credentials are needed.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
lock=stack.lock.json

for tool in jq git curl sha256sum; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }
done
"$repo/scripts/validate-release.sh"

[[ "$(jq -er '.patches | length' "$lock")" == 2 ]] || {
  echo "v0.1.0-rc.44 requires exactly two integration patches" >&2; exit 1;
}
while IFS= read -r patch_path; do
  [[ -f "$repo/$patch_path" ]] || { echo "missing patch: $patch_path" >&2; exit 1; }
done < <(jq -er '.patches[].path' "$lock")
while IFS= read -r patch_path; do
  jq -e --arg path "$patch_path" '.patches[] | select(.path == $path)' "$lock" >/dev/null || {
    echo "patches/ contains unrecorded file: $patch_path" >&2; exit 1;
  }
done < <(find patches -type f -print | sort)

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

verify_patched_tree() {
  local name=$1 repository=$2 head=$3 upstream_tree=$4 result_tree=$5
  local patch_path=$6 patch_sha256=$7 destination="$work/$1"
  git init -q "$destination"
  git -C "$destination" remote add origin "$repository"
  git -C "$destination" fetch -q --depth=1 origin "$head"
  git -C "$destination" checkout -q --detach FETCH_HEAD
  local actual_head actual_upstream_tree actual_patch_sha actual_result_tree
  actual_head=$(git -C "$destination" rev-parse HEAD)
  actual_upstream_tree=$(git -C "$destination" rev-parse 'HEAD^{tree}')
  actual_patch_sha=$(sha256sum "$repo/$patch_path" | cut -d' ' -f1)
  [[ "$actual_head" == "$head" ]] || { echo "$name head mismatch" >&2; exit 1; }
  [[ "$actual_upstream_tree" == "$upstream_tree" ]] || {
    echo "$name upstream tree mismatch" >&2; exit 1;
  }
  [[ "$actual_patch_sha" == "$patch_sha256" ]] || {
    echo "$name patch checksum mismatch" >&2; exit 1;
  }
  git -C "$destination" apply --check "$repo/$patch_path"
  git -C "$destination" apply --index "$repo/$patch_path"
  actual_result_tree=$(git -C "$destination" write-tree)
  [[ "$actual_result_tree" == "$result_tree" ]] || {
    echo "$name result tree mismatch" >&2; exit 1;
  }
  printf '  %s upstream %s tree %s + patch %s -> tree %s\n' \
    "$name" "$actual_head" "$actual_upstream_tree" "$actual_patch_sha" "$actual_result_tree"
}

echo "== source trees =="
verify_patched_tree sglang \
  "$(jq -er '.integration.sglang.repository' "$lock")" \
  "$(pin GLM53_SGLANG_HEAD)" "$(pin GLM53_SGLANG_UPSTREAM_TREE)" \
  "$(pin GLM53_SGLANG_TREE)" \
  "$(jq -er '.integration.sglang.patch' "$lock")" \
  "$(pin GLM53_SGLANG_PATCH_SHA256)"
verify_patched_tree flashinfer \
  "$(jq -er '.integration.flashinfer.repository' "$lock")" \
  "$(pin GLM53_FLASHINFER_HEAD)" "$(pin GLM53_FLASHINFER_UPSTREAM_TREE)" \
  "$(pin GLM53_FLASHINFER_TREE)" \
  "$(jq -er '.integration.flashinfer.patch' "$lock")" \
  "$(pin GLM53_FLASHINFER_PATCH_SHA256)"
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

echo "release bundle verified: official SGLang and FlashInfer bases plus internal patches reproduce exact trees; exact ModelOpt tree"
