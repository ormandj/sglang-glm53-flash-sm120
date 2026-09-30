# Builds and GitHub Releases

The repository uses three separate workflows. A successful image build does not create a GitHub Release.

| Workflow | Result |
|---|---|
| `build-image.yml` | Builds and publishes the immutable candidate image named in `release.json`. |
| `promote-release.yml` | Publishes the stable image tag with the candidate's exact manifest digest. |
| `publish-release.yml` | Verifies the stable image, then creates the source tag and a published GitHub Release with the changelog and image digest. |

Validation runs check source and documentation. They do not publish an image or Release. Image-input pushes to `main` start a candidate build; documentation and workflow changes do not rebuild an existing candidate. Published image and source tags are immutable.

## Publish a new release

Run the release, documentation and source checks described in [RUN.md](RUN.md), and complete hardware qualification before promoting a candidate. Wait for the candidate build to succeed and verify anonymous pull access. If a build fails, inspect its logs before retrying; confirm that the candidate tag is absent before starting another build of that tag.

Dispatch `promote-release.yml` from `main` and wait for success. Verify that the candidate and stable tags resolve to the same digest. Finalize and review the release documentation before selecting the source commit.

Dispatch the repository publisher separately using the full reviewed source SHA derived from Git:

```bash
git switch main
git pull --ff-only
release_target=$(git rev-parse HEAD)
gh workflow run publish-release.yml \
  --repo ormandj/sglang-glm53-flash-sm120 \
  --ref main \
  -f "release_target=$release_target"
```

Wait for this workflow to succeed. It checks the source revision, candidate/stable digest equality and canonical release body before verifying the created source tag and non-draft, non-prerelease Release. A dispatched or running workflow is still pending publication.

## Verify publication

```bash
release_tag=$(jq -er '.stable_tag' release.json)
gh release view "$release_tag" \
  --repo ormandj/sglang-glm53-flash-sm120 \
  --json url,tagName,isDraft,isPrerelease,body
gh api "repos/ormandj/sglang-glm53-flash-sm120/git/ref/tags/$release_tag" \
  --jq '.object'
```

Confirm the Release URL is accessible, its body is the reviewed text, and its tag resolves to the reviewed source commit. For an annotated tag, resolve the tag object to its commit before comparing. Independently confirm anonymous access to the stable container and that its digest matches the candidate. Report release completion only after all of these checks pass. A container tag or Git tag without a published Release does not satisfy this check.
