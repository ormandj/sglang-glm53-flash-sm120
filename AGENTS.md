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

`glm5_next` is not upstream, and the vendor base image has no git provenance.
The lock records `verification.sglang_source_verifiable: false` and
`verification.sglang_repository: null`, and `verify-patches.sh` FAILS if either
is changed to imply otherwise. If you find yourself wanting to assert an SGLang
commit, check whether it actually exists first — do not copy the Qwen build's
tree-hash assertions across.

## Verification before commit

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

The last one needs network: it re-fetches the pinned FlashInfer objects and
re-resolves the base image digests against the registry.

## Claims discipline

Performance and quality numbers belong in `BENCHMARKS.md`, backed by evidence
files, and only when they were actually measured on this exact candidate. A
candidate is "built", not "qualified", until that evidence exists. Do not
promote an rc to stable on the strength of a build succeeding.
