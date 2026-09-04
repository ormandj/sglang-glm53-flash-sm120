#!/usr/bin/env bash
set -euo pipefail

mode=${1:-}
if [[ "$mode" != check && "$mode" != publish ]]; then
  echo "usage: $0 check|publish" >&2
  exit 2
fi

required=(
  RELEASE_PROVIDER
  RELEASE_API_URL
  RELEASE_REPOSITORY
  RELEASE_TOKEN
  RELEASE_TAG
  RELEASE_TARGET
)
if [[ "$mode" == publish ]]; then
  required+=(RELEASE_IMAGE RELEASE_CANDIDATE_IMAGE RELEASE_DIGEST RELEASE_IMAGE_SOURCE_REVISION)
fi
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: $name" >&2; exit 2; }
done

[[ "$RELEASE_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || {
  echo "invalid repository: $RELEASE_REPOSITORY" >&2
  exit 2
}
[[ "$RELEASE_TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "stable release tag must be complete SemVer: $RELEASE_TAG" >&2
  exit 2
}
[[ "$RELEASE_TARGET" =~ ^[0-9a-f]{40}$ ]] || {
  echo "release target must be a full commit SHA: $RELEASE_TARGET" >&2
  exit 2
}
if [[ "$mode" == publish && ! "$RELEASE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "release digest is invalid: $RELEASE_DIGEST" >&2
  exit 2
fi
if [[ "$mode" == publish && ! "$RELEASE_IMAGE_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "image source revision must be a full commit SHA: $RELEASE_IMAGE_SOURCE_REVISION" >&2
  exit 2
fi

api=${RELEASE_API_URL%/}
release_url="${api}/repos/${RELEASE_REPOSITORY}/releases/tags/${RELEASE_TAG}"
create_url="${api}/repos/${RELEASE_REPOSITORY}/releases"
expected_ref="refs/tags/${RELEASE_TAG}"

case "$RELEASE_PROVIDER" in
  forgejo)
    ref_url="${api}/repos/${RELEASE_REPOSITORY}/git/refs/tags/${RELEASE_TAG}"
    auth_header="Authorization: token ${RELEASE_TOKEN}"
    ;;
  github)
    ref_url="${api}/repos/${RELEASE_REPOSITORY}/git/ref/tags/${RELEASE_TAG}"
    auth_header="Authorization: Bearer ${RELEASE_TOKEN}"
    ;;
  *)
    echo "unsupported release provider: $RELEASE_PROVIDER" >&2
    exit 2
    ;;
esac

release_tmp=$(mktemp -d)
trap 'rm -rf "$release_tmp"' EXIT

api_request() {
  local method=$1 url=$2 output=$3 data_file=${4:-}
  local -a args=(
    --silent
    --show-error
    --request "$method"
    --header "$auth_header"
    --header 'Accept: application/json'
    --output "$output"
    --write-out '%{http_code}'
  )
  if [[ "$RELEASE_PROVIDER" == github ]]; then
    args+=(--header 'X-GitHub-Api-Version: 2022-11-28')
  fi
  if [[ -n "$data_file" ]]; then
    args+=(--header 'Content-Type: application/json' --data-binary "@$data_file")
  fi
  curl "${args[@]}" "$url"
}

show_api_error() {
  local response=$1
  jq -r '.message // .error // "unknown API error"' "$response" >&2 2>/dev/null \
    || echo "unknown API error" >&2
}

load_state() {
  local ref_status release_status

  ref_status=$(api_request GET "$ref_url" "$release_tmp/ref.json")
  case "$ref_status" in
    200)
      if [[ "$RELEASE_PROVIDER" == forgejo ]]; then
        release_ref_sha=$(jq -r --arg ref "$expected_ref" '
          if type == "array" then
            first(.[] | select(.ref == $ref) | .object.sha) // empty
          elif .ref == $ref then
            .object.sha // empty
          else
            empty
          end
        ' "$release_tmp/ref.json")
        release_ref_type=$(jq -r --arg ref "$expected_ref" '
          if type == "array" then
            first(.[] | select(.ref == $ref) | .object.type) // empty
          elif .ref == $ref then
            .object.type // empty
          else
            empty
          end
        ' "$release_tmp/ref.json")
      else
        release_ref_sha=$(jq -r --arg ref "$expected_ref" '
          if .ref == $ref then .object.sha // empty else empty end
        ' "$release_tmp/ref.json")
        release_ref_type=$(jq -r --arg ref "$expected_ref" '
          if .ref == $ref then .object.type // empty else empty end
        ' "$release_tmp/ref.json")
      fi
      ;;
    404)
      release_ref_sha=
      release_ref_type=
      ;;
    *)
      echo "release-tag lookup failed with HTTP $ref_status" >&2
      show_api_error "$release_tmp/ref.json"
      exit 1
      ;;
  esac

  if [[ -n "$release_ref_sha" ]]; then
    local depth tag_status
    for depth in {1..8}; do
      case "$release_ref_type" in
        commit) break ;;
        tag)
          tag_status=$(api_request GET \
            "${api}/repos/${RELEASE_REPOSITORY}/git/tags/${release_ref_sha}" \
            "$release_tmp/tag-${depth}.json")
          if [[ "$tag_status" != 200 ]]; then
            echo "annotated-tag lookup failed with HTTP $tag_status" >&2
            show_api_error "$release_tmp/tag-${depth}.json"
            exit 1
          fi
          release_ref_sha=$(jq -er '.object.sha' "$release_tmp/tag-${depth}.json")
          release_ref_type=$(jq -er '.object.type' "$release_tmp/tag-${depth}.json")
          ;;
        *)
          echo "$expected_ref targets unsupported Git object type: $release_ref_type" >&2
          exit 1
          ;;
      esac
    done
    [[ "$release_ref_type" == commit ]] || {
      echo "$expected_ref did not peel to a commit" >&2
      exit 1
    }
  fi

  release_status=$(api_request GET "$release_url" "$release_tmp/release.json")
  case "$release_status" in
    200) release_exists=1 ;;
    404) release_exists=0 ;;
    *)
      echo "release lookup failed with HTTP $release_status" >&2
      show_api_error "$release_tmp/release.json"
      exit 1
      ;;
  esac
}

validate_state() {
  if [[ -n "$release_ref_sha" && "$release_ref_sha" != "$RELEASE_TARGET" ]]; then
    echo "$expected_ref already targets $release_ref_sha, expected $RELEASE_TARGET" >&2
    exit 1
  fi
  if (( release_exists )) && [[ -z "$release_ref_sha" ]]; then
    echo "release $RELEASE_TAG exists without its expected Git ref" >&2
    exit 1
  fi
  if (( release_exists )); then
    jq -e --arg tag "$RELEASE_TAG" '
      .tag_name == $tag and .name == $tag and .draft == false and .prerelease == false
    ' "$release_tmp/release.json" >/dev/null || {
      echo "release $RELEASE_TAG exists with unexpected metadata" >&2
      exit 1
    }
  fi
}

load_state
validate_state
release_was_existing=$release_exists

if [[ "$mode" == check ]]; then
  if (( release_exists )); then
    echo "release $RELEASE_TAG already exists at $RELEASE_TARGET"
  elif [[ -n "$release_ref_sha" ]]; then
    echo "$expected_ref exists at $RELEASE_TARGET; its Release can be resumed"
  else
    echo "release name and tag are available: $RELEASE_TAG"
  fi
  exit 0
fi

{
  printf '# %s\n\n' "$RELEASE_TAG"
  printf 'Image: `%s`, digest `%s`.\n\n' "$RELEASE_IMAGE" "$RELEASE_DIGEST"
  printf 'This is a digest-identical promotion of `%s` within this registry.\n\n' "$RELEASE_CANDIDATE_IMAGE"
  printf 'Image source revision: `%s`.\n\n' "$RELEASE_IMAGE_SOURCE_REVISION"
  printf 'Release tag target: `%s`.\n\n' "$RELEASE_TARGET"
  printf 'Detailed changes are recorded in `CHANGELOG.md`; measured qualification evidence is maintained in the primary project repository.\n'
} >"$release_tmp/body.md"

if (( ! release_exists )); then
  jq -n \
    --arg tag "$RELEASE_TAG" \
    --arg target "$RELEASE_TARGET" \
    --rawfile body "$release_tmp/body.md" \
    '{
      tag_name: $tag,
      target_commitish: $target,
      name: $tag,
      body: $body,
      draft: false,
      prerelease: false
    }' >"$release_tmp/create.json"

  create_status=$(api_request POST "$create_url" "$release_tmp/create-response.json" "$release_tmp/create.json")
  case "$create_status" in
    201) ;;
    409|422)
      release_was_existing=1
      ;;
    *)
      echo "release creation failed with HTTP $create_status" >&2
      show_api_error "$release_tmp/create-response.json"
      exit 1
      ;;
  esac
fi

load_state
validate_state
[[ "$release_ref_sha" == "$RELEASE_TARGET" ]] || {
  echo "$expected_ref was not created at $RELEASE_TARGET" >&2
  exit 1
}
if (( ! release_exists )); then
  echo "release $RELEASE_TAG was not created; an inaccessible draft may own the tag" >&2
  if [[ -s "$release_tmp/create-response.json" ]]; then
    show_api_error "$release_tmp/create-response.json"
  fi
  exit 1
fi
jq -rj '.body' "$release_tmp/release.json" >"$release_tmp/actual-body.md"
cmp -s "$release_tmp/body.md" "$release_tmp/actual-body.md" || {
  echo "release $RELEASE_TAG exists with a noncanonical body" >&2
  exit 1
}
if (( release_was_existing )); then
  echo "verified existing release $RELEASE_TAG at $RELEASE_TARGET for $RELEASE_IMAGE@$RELEASE_DIGEST"
else
  echo "published release $RELEASE_TAG at $RELEASE_TARGET for $RELEASE_IMAGE@$RELEASE_DIGEST"
fi
