# Working in this repository

This repository builds ONE immutable container candidate at a time. The rules
below exist because published tags are immutable and cannot be corrected.

## Release naming

Every reference to the image — README, RUN.md, CHANGELOG, launcher default,
workflows — always uses the complete release name, never an abbreviation and
never a floating tag. `scripts/validate-docs.sh` enforces this.

Bump `release.json` `candidate` for any change to `Containerfile`,
`release.json`, `stack.lock.json`, or `patches/**`. Those paths are exactly what
the build workflow triggers on; anything else cannot replace a published tag.

## Do not overstate provenance

The vendor base image still has no SGLang git provenance and must be described
only as the pinned CUDA/PyTorch dependency stack. The active SGLang Python tree
is produced from the exact official-upstream commit plus the checksummed
integration patch recorded in `stack.lock.json`; `verify-patches.sh` must
reproduce the final tree. The internal working-fork commit is provenance for
the editable branch, not a claim that the integration commit exists on
`sgl-project/sglang` upstream. Upstream-main base commits and PR numbers are
ancestry/context, not provenance for project-owned changes.

## Verification before commit

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

The last one needs network: it re-fetches the pinned official SGLang,
FlashInfer, and ModelOpt objects, applies the recorded internal patches, and
re-resolves the base image digests against the registry.

## Claims discipline

Do not add performance or quality numbers to this repository. Measured results
belong in the primary `sglang-glm53-flash-sm120` repository, backed by evidence
files. A candidate is "built", not "qualified", until that evidence exists.

## Release flow: internal first, promote and publish by dispatch

Candidates are built and validated internally (Forgejo, deployed to quasar via
gitops) before anything is published to GitHub/ghcr. Stable SemVer tags are
created ONLY by `promote-release.yml`, which is `workflow_dispatch`-only — a
push cannot trigger it; dispatch it from the Forgejo Actions UI. It reads
`candidate_tag`/`stable_tag` from `release.json` on `main`, refuses if the
stable tag already exists, performs a digest-identical `skopeo copy`, and
verifies the digest did not change. skopeo source references must be
digest-only (`repo@sha256:...`); it rejects `tag@digest` combined references.
After promotion, verify the stable tag resolves to the candidate digest
(`docker buildx imagetools inspect`) before updating any docs or announcing.
After the verified stable-image promotion, update the stable README and
CHANGELOG on `main`, then dispatch `publish-release.yml`. That workflow
re-verifies the candidate/stable digest equality, refuses a conflicting source
tag or Release, and creates the repository Release and its source tag together
at the exact stable-doc commit. Pass the same full commit SHA as the required
`release_target` input to Forgejo first and GitHub second; each workflow fails
unless its checked-out `main` is that commit. Do not push stable source tags
manually.

## External contributions (the GitHub mirror)

Issues and PRs arrive on `github.com/ormandj/sglang-glm53-flash-sm120`.

- Review a PR by materializing it: apply its
  `patches/sglang-glm53-integration.patch` to a clean checkout of the pinned
  upstream commit and diff that tree against ours. The PR's real delta is that
  diff — the patch-of-patch in the git diff is unreadable and hides scope.
  Compare both modified AND newly-added files (`git diff` misses untracked
  additions).
- Run all three validation scripts against the PR branch in a worktree, and
  confirm the recorded patch sha256 matches the actual patch file.
- Verify every claim in the PR text against the code. Claimed fixes have been
  absent from the diff before; a claim is not evidence.
- Two independent reviews before merge (a line-level read plus a second
  model's opinion), and on-our-hardware validation (differential oracle,
  sanitizer run, the original reproducer). External test results are
  corroboration, not a substitute.
- Require rebase onto current `main` before merge — stale forks silently
  revert launcher/docs state — and renumber the contribution as the next
  release candidate.
- Credit external reporters and contributors by name in CHANGELOG and release
  notes.
- The DFlash2 drafter (incoai, CC BY-NC-ND 4.0) must never enter this
  repository or any published image.
- Write issue comments and reviews in the maintainer's own voice, technically
  substantive, with no assistance attributions.
