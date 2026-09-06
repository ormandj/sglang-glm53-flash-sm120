#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
stable_tag=$(jq -er '.stable_tag' "$repo/release.json")
provider=${RELEASE_PROVIDER:-github}

require_text() {
  local file=$1 expected=$2
  grep -F -- "$expected" "$file" >/dev/null || {
    echo "${file#$repo/} missing stable-release text: $expected" >&2
    exit 1
  }
}

case "$provider" in
  forgejo)
    # Registry identity is verified by the promotion/publication workflow.
    # Public documentation must not contain private registry status.
    ;;
  github)
    public_image="ghcr.io/ormandj/sglang-glm53-flash-sm120:${stable_tag}"
    require_text "$repo/README.md" "| Image | \`${public_image}\` |"
    require_text "$repo/README.md" "The current published stable image is \`${stable_tag}\`"
    ;;
  *) echo "unsupported release provider: $provider" >&2; exit 2 ;;
esac
require_text "$repo/CHANGELOG.md" "## ${stable_tag} (stable;"
python3 "$repo/scripts/render-release-changes.py" \
  --changelog "$repo/CHANGELOG.md" --tag "$stable_tag" >/dev/null

echo "stable publication docs valid: ${provider} ${stable_tag}"
