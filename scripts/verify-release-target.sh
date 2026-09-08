#!/usr/bin/env bash
# Permit publication-tool repairs without changing an immutable release target.
set -euo pipefail
release_target=${1:?full release commit required}
workflow_revision=${2:?full workflow commit required}
[[ "$release_target" =~ ^[0-9a-f]{40}$ ]]
[[ "$workflow_revision" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git rev-parse HEAD)" == "$workflow_revision" ]]
[[ "$(git rev-parse "${release_target}^{commit}")" == "$release_target" ]]
git merge-base --is-ancestor "$release_target" "$workflow_revision"
changed_files=$(mktemp)
trap 'rm -f "$changed_files"' EXIT
git diff --name-only --no-renames -z "$release_target" "$workflow_revision" >"$changed_files"
while IFS= read -r -d '' changed_path; do
  case "$changed_path" in
    .github/workflows/publish-release.yml|.forgejo/workflows/publish-release.yml|scripts/verify-release-target.sh|scripts/test_verify_release_target.py)
      ;;
    *)
      printf 'release target differs outside publication tooling: %s\n' "$changed_path" >&2
      exit 1
      ;;
  esac
done <"$changed_files"
printf 'verified release target: %s\n' "$release_target"
