#!/usr/bin/env bash
# Verify everything about this release bundle that is ACTUALLY verifiable.
#
# The sibling sglang-qwen38-flash-next-sm120 build reproduces the exact SGLang
# source tree: it pins main by commit and tree, applies an archived patch, and
# asserts the resulting effective tree hash. That is deliberately NOT done here,
# because it cannot be:
#
#   * `glm5_next` is not in sgl-project/sglang main as of 2026-08-27. The
#     architecture exists only inside the vendor per-model image.
#   * That image is built from `ADD sglang.tar.gz`, not a git checkout, and
#     reports SGLANG_BUILD_COMMIT=unknown. There is no commit or tree to check.
#
# So this script verifies: the release contract, that every Containerfile pin is
# recorded in the lock and vice versa, that archived patch bytes match, that the
# pinned FlashInfer commit and tree really do reproduce, and that the pinned base
# image digests still resolve in the registry. It explicitly refuses to imply
# SGLang source provenance we do not have.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
lock=stack.lock.json
containerfile=Containerfile

for tool in jq git curl; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }
done
if command -v sha256sum >/dev/null; then
  sha256_file() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then
  sha256_file() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "missing SHA-256 tool" >&2; exit 1
fi

"$repo/scripts/validate-release.sh"
lock_value() { jq -r --arg arg "$1" '.pins[] | select(.arg == $arg) | .value' "$lock"; }

echo "== lock and Containerfile pins =="
while IFS=$'\t' read -r arg value; do
  grep -Fxq "ARG ${arg}=${value}" "$containerfile" || { echo "Containerfile pin mismatch: $arg" >&2; exit 1; }
  printf '  %s\n' "$arg"
done < <(jq -r '.pins[] | [.arg, .value] | @tsv' "$lock")
while IFS= read -r arg; do
  [[ -n "$(lock_value "$arg")" ]] || { echo "unrecorded Containerfile pin: $arg" >&2; exit 1; }
done < <(sed -n 's/^ARG \([A-Z0-9_]*\)=.*/\1/p' "$containerfile")

echo "== archived patch bytes =="
patch_count=$(jq -r '.patches | length' "$lock")
if [[ "$patch_count" == 0 ]]; then
  # No patches at rc.1. Assert the tree really is empty so a dropped patch
  # cannot pass silently.
  if [[ -d patches ]] && [[ -n "$(find patches -type f -print -quit)" ]]; then
    echo "patches/ contains files but stack.lock.json records none" >&2; exit 1
  fi
  echo "  (none recorded; patches/ is empty)"
else
  while IFS=$'\t' read -r path sha; do
    [[ -f "$path" ]] || { echo "missing patch: $path" >&2; exit 1; }
    [[ "$(sha256_file "$path")" == "$sha" ]] || { echo "SHA-256 mismatch: $path" >&2; exit 1; }
    grep -Fq "COPY ${path} " "$containerfile" || { echo "Containerfile does not copy $path" >&2; exit 1; }
    printf '  %s\n' "$path"
  done < <(jq -r '.patches[] | [.path, .sha256] | @tsv' "$lock")
  while IFS= read -r path; do
    [[ "$(jq -r --arg p "$path" '[.patches[] | select(.path == $p)] | length' "$lock")" == 1 ]] || {
      echo "unrecorded patch: $path" >&2; exit 1; }
  done < <(find patches -type f | sort)
fi

echo "== SGLang provenance disclosure =="
if [[ "$(jq -r '.verification.sglang_source_verifiable' "$lock")" != "false" ]]; then
  echo "lock claims SGLang source is verifiable; it is not for a vendor tarball image" >&2
  exit 1
fi
if [[ "$(jq -r '.verification.sglang_repository' "$lock")" != "null" ]]; then
  echo "lock names an SGLang repository but the base carries no commit to check" >&2
  exit 1
fi
printf '  base pinned by digest only: %s\n' "$(lock_value GLM53_SGLANG_BASE)"
printf '  linux/amd64 manifest:       %s\n' "$(lock_value GLM53_SGLANG_BASE_AMD64_MANIFEST)"

echo "== FlashInfer source tree =="
fi_repo=$(jq -er '.verification.flashinfer_repository' "$lock")
fi_head=$(lock_value GLM53_FLASHINFER_MAIN_HEAD)
fi_tree=$(lock_value GLM53_FLASHINFER_MAIN_TREE)
[[ "$(jq -er '.integration.flashinfer.head' "$lock")" == "$fi_head" ]] || { echo "FlashInfer head mismatch" >&2; exit 1; }
[[ "$(jq -er '.integration.flashinfer.tree' "$lock")" == "$fi_tree" ]] || { echo "FlashInfer tree mismatch" >&2; exit 1; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init -q "$work/flashinfer"
git -C "$work/flashinfer" remote add origin "$fi_repo"
git -C "$work/flashinfer" fetch -q --depth=1 origin "$fi_head"
git -C "$work/flashinfer" checkout -q --detach FETCH_HEAD
actual_head=$(git -C "$work/flashinfer" rev-parse HEAD)
actual_tree=$(git -C "$work/flashinfer" rev-parse 'HEAD^{tree}')
[[ "$actual_head" == "$fi_head" ]] || { echo "FlashInfer head did not reproduce: $actual_head" >&2; exit 1; }
[[ "$actual_tree" == "$fi_tree" ]] || { echo "FlashInfer tree did not reproduce: $actual_tree" >&2; exit 1; }
printf '  head %s\n  tree %s\n' "$actual_head" "$actual_tree"

echo "== base image digests resolve =="
base_ref=$(lock_value GLM53_SGLANG_BASE)
index_digest=${base_ref##*@}
amd64_digest=$(lock_value GLM53_SGLANG_BASE_AMD64_MANIFEST)
token=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:lmsysorg/sglang:pull" | jq -er .token)
accept='application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json'
for digest in "$index_digest" "$amd64_digest"; do
  curl -fsSL -o /dev/null -H "Authorization: Bearer ${token}" -H "Accept: ${accept}" \
    "https://registry-1.docker.io/v2/lmsysorg/sglang/manifests/${digest}" || {
      echo "base digest does not resolve: $digest" >&2; exit 1; }
  printf '  %s\n' "$digest"
done

echo "release bundle verified (SGLang layer pinned by digest; source not reproducible by design)"
