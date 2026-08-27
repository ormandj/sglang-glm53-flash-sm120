#!/usr/bin/env bash
set -euo pipefail

forgejo_url=${FORGEJO_URL:-https://git.home.corenode.com}
package_owner=${PACKAGE_OWNER:-homelab}
package_name=${PACKAGE_NAME:-sglang-glm53-flash-sm120-container}
keep_releases=${KEEP_RELEASES:-6}
apply=false

usage() {
  cat <<'EOF'
Usage: prune-container-releases.sh [--apply]

Keep the newest SemVer release tags and their recent numeric build aliases for
one Forgejo container package. The default is a dry run.

Environment:
  FORGEJO_TOKEN  Required Forgejo token with package read/delete access
  FORGEJO_URL    Forgejo base URL (default: https://git.home.corenode.com)
  PACKAGE_OWNER  Package owner (default: homelab)
  PACKAGE_NAME   Container package name (default: sglang-glm53-flash-sm120-container)
  KEEP_RELEASES  Number of release tags and build aliases to keep (default: 6)
EOF
}

while (($#)); do
  case "$1" in
    --apply)
      apply=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

for tool in curl cut jq sort; do
  command -v "$tool" >/dev/null || {
    echo "missing required tool: $tool" >&2
    exit 1
  }
done

[[ -n "${FORGEJO_TOKEN:-}" ]] || {
  echo "FORGEJO_TOKEN is required" >&2
  exit 1
}
[[ "$keep_releases" =~ ^[1-9][0-9]*$ ]] || {
  echo "KEEP_RELEASES must be a positive integer" >&2
  exit 1
}

inventory=$(mktemp)
trap 'rm -f "$inventory"' EXIT
printf '[]\n' >"$inventory"

page=1
limit=50
while :; do
  response=$(curl --fail --silent --show-error \
    --header "Authorization: token ${FORGEJO_TOKEN}" \
    --get \
    --data-urlencode 'type=container' \
    --data-urlencode "q=${package_name}" \
    --data-urlencode "limit=${limit}" \
    --data-urlencode "page=${page}" \
    "${forgejo_url}/api/v1/packages/${package_owner}")
  count=$(jq -er 'length' <<<"$response")
  jq --argjson page "$response" '. + $page' "$inventory" >"${inventory}.next"
  mv "${inventory}.next" "$inventory"
  ((count < limit)) && break
  ((page += 1))
done

mapfile -t all_versions < <(
  jq -r --arg package "$package_name" \
    '.[] | select(.name == $package) | .version' "$inventory" | sort -u
)
mapfile -t release_versions < <(
  printf '%s\n' "${all_versions[@]}" \
    | jq -Rr 'select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+(-rc\\.[0-9]+)?$"))' \
    | while IFS= read -r version; do
        if [[ "$version" == *-rc.* ]]; then
          printf '%s\t%s\n' "$version" "$version"
        else
          printf '%s-zz\t%s\n' "$version" "$version"
        fi
      done \
    | sort -k1,1Vr \
    | cut -f2
)
mapfile -t build_versions < <(
  printf '%s\n' "${all_versions[@]}" \
    | jq -Rr 'select(test("^build-[0-9]+$"))' \
    | sort -t- -k2,2nr
)

declare -A keep=()
for version in "${release_versions[@]:0:keep_releases}"; do
  keep["$version"]=1
done
for version in "${build_versions[@]:0:keep_releases}"; do
  keep["$version"]=1
done

delete_versions=()
for version in "${all_versions[@]}"; do
  [[ -n "${keep[$version]:-}" ]] || delete_versions+=("$version")
done

echo "package: ${package_owner}/${package_name}"
echo "keeping ${#keep[@]} tag(s):"
printf '  %s\n' "${!keep[@]}" | sort -V

if ((${#delete_versions[@]} == 0)); then
  echo "nothing to delete"
  exit 0
fi

if [[ "$apply" != true ]]; then
  echo "dry run; would delete ${#delete_versions[@]} tag(s):"
  printf '  %s\n' "${delete_versions[@]}"
  exit 0
fi

echo "deleting ${#delete_versions[@]} tag(s):"
for version in "${delete_versions[@]}"; do
  echo "  ${version}"
  curl --fail --silent --show-error \
    --request DELETE \
    --header "Authorization: token ${FORGEJO_TOKEN}" \
    "${forgejo_url}/api/v1/packages/${package_owner}/container/${package_name}/${version}"
done
