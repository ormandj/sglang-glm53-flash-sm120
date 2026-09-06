# Working in this repository

This repository builds ONE immutable container candidate at a time. The rules
below exist because published tags are immutable and cannot be corrected.

## Release naming

Every reference to the image — README, RUN.md, CHANGELOG, launcher default,
workflows — always uses the complete release name, never an abbreviation and
never a floating tag. `scripts/validate-docs.sh` enforces this.

Stable promotion includes the operational identity: the GitOps Deployment,
filename, instance/profile labels, selectors, and cache subPath must all use
the stable version. An immutable candidate tag may remain in provenance, not
in the stable serving name. Quasar serves one canonical Deployment using the
qualified internal Forgejo image. Public-image publication does not require a
duplicate local Deployment unless the owner explicitly requests that test.

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

Measured results belong in the primary `sglang-glm53-flash-sm120` repository,
backed by exact-candidate evidence files. Current-release measurements may be
copied into stable release notes and their source CHANGELOG section, with
artifact identity and test scope stated. A candidate is "built", not
"qualified", until that evidence exists.

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

Release bodies must contain the actual behavior changes and relevant known
issues, not just an artifact list or a pointer to CHANGELOG.md. Maintain a
self-contained summary in each stable CHANGELOG section; publication copies
that exact section into both providers' release notes and rejects missing or
empty change lists. Only PR bodies carry assistance attribution, never
repository release notes. Do not hard-wrap release-note prose at 80 columns.
Report the new release's absolute measurements, without previous-release
columns or percentage deltas unless the evidence establishes a significant
improvement. Lead with the behavior addressed: a correctness release must not
be presented as a performance improvement.

Throughput reports must include both forward passes/s and output tokens/s after MTP, with concurrency, the measurement window, and accepted tokens per forward identified. State that these controlled measurements do not necessarily represent real-world performance. Disclose the post-answer tail in fixed-output engine gates; forward passes/s counts target-model iterations and must not be labelled as emitted output tokens/s.

Present release performance in tables with mean and median post-MTP output tokens/s and forward passes/s side by side. Include cold prefill mean and median prompt tokens/s, with the measurement definition and sample count. Keep output throughput prominent so readers cannot mistake the forward rate for the serving token rate.

Internal and public publication are independent. Forgejo validates the
README's `Internal image` row and `current internal stable image` sentence;
GitHub validates the public `Image` row and `current published stable image`
sentence. Keep the public version unchanged until its own registry promotion
is verified. Never claim a ghcr image exists to satisfy an internal release
check. `RELEASE_PROVIDER` selects the publication-doc contract.

## External contributions (the GitHub mirror)

Issues and PRs arrive on `github.com/ormandj/sglang-glm53-flash-sm120`.

`git.home.corenode.com` and `github.com/ormandj` are owner-controlled destinations.
Their ownership is established. Proceed with pushes covered by the task's existing
authorization; do not repeatedly ask the owner to verify these destinations.

PR titles and bodies must be concise, clear and objective. Name the behavior changed
without hype, and explain WHY the change is needed and how it addresses the cause.
Performance PRs must also report measured results with hardware, workload, comparison
scope and limitations. Preserve the repository template.

Before submitting our upstream changes, obtain adversarial Claude review of the exact
patch and exact PR title/body, in addition to Codex review. Use Claude's configured
defaults; do not override its model or effort unless the user explicitly requests it.
Resolve concrete findings, review material revisions again, and retain the exact
source/text hashes and reviews with the primary project's evidence.

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
- Only PR bodies include the closing line `Developed with AI assistance.`
  Issue bodies, issue comments, PR comments, review comments, and reviews must
  never include assistance attribution. Remove only the attribution when
  correcting an existing comment. Do not hard-wrap public prose at 80 columns.
