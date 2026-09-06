# Working in this repository

This is the public-facing source and container repository for GLM-5.3-Flash on SM120. Documentation is written for users and contributors.

## Public content

Internal housekeeping NEVER belongs in this repository, including README, CHANGELOG, release notes, issues, PR text, AGENTS.md, or separate status/evidence files. Do not include private registry addresses or digests, homelab host/cluster/deployment names, private checkout paths, candidate/build/promotion bookkeeping, operational instructions, or internal review/test-run narration. Keep those records in the private qualification project. Moving them to another public file does not resolve the problem.

README describes the current published image, requirements, installation, usage, measured behavior and actionable limitations. Link to the changelog and published Releases for history. Do not include a candidate-by-candidate release ledger or failed-build history.

Release notes describe changes actually delivered and their effect on users, with relevant upgrade instructions and known limitations. Do not copy internal test notes, grader/oracle diagnostics, or instructions to perform an already completed promotion. Review the exact generated body before publication. Publishing templates and checks must not require internal status in public documentation.

Public source provenance and upstream contribution details must be factual and useful to contributors. Keep private coordination logs and branch-maintenance bookkeeping in the private project.

## Release integrity

Every reference to the image always uses the complete release name, never an abbreviation or a floating tag. Published image and source tags are immutable. Documentation corrections must not move existing tags or rebuild an existing candidate.

Bump release.json candidate for changes to Containerfile, release.json, stack.lock.json or patches/**. Those are the image build inputs. A successful build alone does not qualify a release. Stable images and repository releases are created through the release workflows, with registry digest equality verified before publication.

The vendor base supplies the pinned CUDA/PyTorch dependency stack and does not establish SGLang source provenance. stack.lock.json and scripts/verify-patches.sh define the official source bases, checksummed integration patches and resulting trees. A fork integration commit is not an upstream commit. Preserve historical pins.

## Verification before commit

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

The last check needs network access to verify pinned source trees and base image digests. For publication tooling changes, also run python3 scripts/test_validate_publish_release.py and validate each provider's publication contract.

## Measurements

Performance and quality claims require measured evidence for the stated artifact, hardware, configuration and workload. If measurements use a validation build produced separately from the public image, say so plainly. Do not imply the public image was tested. Keep detailed private artifact bindings and operational receipts in the private qualification project.

Report mean and median output tokens/s after MTP alongside target forward passes/s, with concurrency, measurement window and accepted output tokens per forward. Report cold prefill mean and median prompt tokens/s, its measurement definition and sample count. Distinguish naturally completed responses from output-capped responses, and disclose the post-answer tail in fixed-output gates. These controlled measurements do not necessarily represent real-world performance.

Describe user-visible limitations in public prose. Detailed grader differences and internal test-oracle results belong in qualification records; do not turn them into a public release narrative.

## Contributions

Preserve contributor credit and the repository PR template. Titles and bodies must clearly describe the behavior changed, why the change is needed and how it addresses the cause. Performance claims need hardware, workload, comparison scope and limitations. Do not hard-wrap public prose or use em dashes.

Only PR bodies include the closing line "Developed with AI assistance." Never add assistance attribution to issues, comments, reviews or release notes. Preserve CI-managed PR footers.

Before submitting upstream patches, obtain Codex and adversarial Claude review of the exact patch and title/body. Use Claude's configured model and effort defaults. Resolve concrete findings and review material changes again. Keep internal review receipts privately. Upstream posts require explicit authorization.

The DFlash2 drafter must not enter this repository or its published images.
