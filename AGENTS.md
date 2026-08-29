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
