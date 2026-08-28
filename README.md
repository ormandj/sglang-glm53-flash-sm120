# SGLang GLM-5.3-Flash SM120 container

Internal immutable build for serving the BF16-derived MXFP4A16 GLM-5.3-Flash
artifact on two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120) at TP=2, with vision
and native MTP retained.

Candidate under construction:
`git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.14`

The primary `sglang-glm53-flash-sm120` repository owns the quantization audit
and all future performance/quality evidence. This build repository makes no
qualification claim.

## Image contents

- Vendor `lmsysorg/sglang:glm-5.3-flash-amd64` base pinned by immutable OCI
  index and linux/amd64 manifest digests.
- Exact byte-gated fixes for GLM's SM120 MXFP4 gate/up and activation contract,
  plus upstream PR #36708's DFlash2 hidden-state capture change and PR #36755's
  mHC `residual=None` correction, and a native GLM-5.3 architecture extension
  to upstream PR #26928's fail-closed SM120 sparse-MLA guard.
- Exact draft PR #36745 fused KPool top-k correction after the vendor preimage
  failed all four deterministic/order/overflow GPU regressions on SM120.
- GLM47-specific backport of merged SGLang #36626 for nested streaming JSON
  closure and top-level composite tool schemas.
- An exact 528-byte GLM-5.3 no-RoPE adapter and FlashInfer SM120 kernel
  specialization that preserve all 2,051 KPool entries by padding with `-1`
  to one 2,176-wide TP=2 dispatch.
- An exact 2,051-entry unfused decode/prefill transform for isolating fused
  KPool correctness without dropping the model's three live tail entries.
- A byte-gated correction that makes GLM's dedicated no-RoPE KV scatter skip
  the reserved padding slot and honor DCP ownership like the ordinary writer.
- A byte-gated SM120/SM121 TileLang launch for the independent BF16-KV no-RoPE
  DSA path, with non-SM12 CUDA and HIP launches left unchanged.
- The SM120 MXFP4 post-loader reads its runner configuration from the
  quantization method that owns it, preserving the vendor GPT-OSS contract as
  well as GLM's distinct gate/up and activation semantics.
- Build-time semantic tests for the runtime patches, including the real mHC
  `residual=None` capture path.
- SM120-safe mHC/indexer settings matching SGLang's current DeepSeek-V4 guard;
  GLM's shared DSA hook does not yet inherit that guard upstream.
- FlashInfer 0.6.18 rebuilt from exact commit
  `93f4f2642e1b3680a52ebb51cf68e0fdad237796` and tree
  `7e9829d1b743896617fbba8ad7d36f3d72127b7e` with
  `FLASHINFER_CUDA_ARCH_LIST=12.0f`; generic cross-architecture cubin and JIT
  cache packages are absent, and runtime artifact downloads are disabled.
- Runtime profiles for verifier-only, native fixed five-step MTP, and pinned
  DFlash2; multimodal execution remains explicitly enabled in every mode.
- An exhaustive compressed-tensors ignore keeps ordinary linears BF16, and
  shared-expert fusion is disabled so its protected BF16 weights never enter
  the MXFP4 routed-expert buffer.

## Provenance boundary

`glm5_next` is not in sgl-project/sglang main as of 2026-08-28, and the vendor
per-model image was built from a tarball with no verifiable SGLang commit. The
lock therefore keeps `verification.sglang_source_verifiable: false` and
`verification.sglang_repository: null`.

The base is reproducible by image digest, not by a claimed SGLang git tree. The
the modified files are separately reproducible by exact preimage SHA-256,
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
`v0.1.0-rc.14` built, not qualified.
