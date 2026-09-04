#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stable_tag=$(jq -er '.stable_tag' "$repo/release.json")
public_image="ghcr.io/ormandj/sglang-glm53-flash-sm120:${stable_tag}"

require_text() {
  local file=$1 expected=$2
  grep -F -- "$expected" "$file" >/dev/null || {
    echo "${file#$repo/} missing stable-release text: $expected" >&2
    exit 1
  }
}

require_text "$repo/README.md" "| Image | \`${public_image}\` |"
require_text "$repo/README.md" "The current published stable image is \`${stable_tag}\`"
require_text "$repo/CHANGELOG.md" "## ${stable_tag} (stable;"

echo "stable publication docs valid: ${stable_tag}"
