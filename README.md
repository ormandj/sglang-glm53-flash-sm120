# SGLang GLM-5.3-Flash SM120 container

Internal immutable build for serving the BF16-derived MXFP4A16 GLM-5.3-Flash
artifact on two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120) at TP=2, with vision
and native MTP retained.

Candidate under construction:
`git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.3`

The primary `sglang-glm53-flash-sm120` repository owns the quantization audit
and all future performance/quality evidence. This build repository makes no
qualification claim.

## Image contents

- Vendor `lmsysorg/sglang:glm-5.3-flash-amd64` base pinned by immutable OCI
  index and linux/amd64 manifest digests.
- Exact byte-gated fixes for GLM's SM120 MXFP4 gate/up and activation contract,
  plus upstream PR #36708's DFlash2 hidden-state capture change.
- Build-time semantic tests for both patches.
- FlashInfer 0.6.18 rebuilt from exact commit
  `cbcbce48e817c83f03ad5a3e6ce59480eaf6935d` and tree
  `d3a639d6f268b8bfc679a8bd15581a6a6b319a16` with
  `FLASHINFER_CUDA_ARCH_LIST=12.0f`.
- Runtime profiles for verifier-only, native adaptive MTP, and pinned DFlash2;
  multimodal execution remains explicitly enabled in every mode.
- An exhaustive compressed-tensors ignore keeps ordinary linears BF16, and
  shared-expert fusion is disabled so its protected BF16 weights never enter
  the MXFP4 routed-expert buffer.

## Provenance boundary

`glm5_next` is not in sgl-project/sglang main as of 2026-08-27, and the vendor
per-model image was built from a tarball with no verifiable SGLang commit. The
lock therefore keeps `verification.sglang_source_verifiable: false` and
`verification.sglang_repository: null`.

The base is reproducible by image digest, not by a claimed SGLang git tree. The
two modified files are separately reproducible by exact preimage SHA-256,
archived patch bytes applied with zero fuzz, and exact postimage SHA-256.

## Verify

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

The final command needs network access to reproduce the pinned FlashInfer tree
and re-resolve the vendor image digests.

## Scope

SM120 and linux/amd64 only. A successful workflow makes
`v0.1.0-rc.3` built, not qualified.
