# SGLang GLM-5.3-Flash SM120 (container build)

Internal container build for serving an **MXFP4A16 quantization of
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)** on two
NVIDIA RTX PRO 6000 Blackwell GPUs (SM120) at TP=2.

Published candidate: `git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.1`

The public source-release counterpart, including the full quantization rationale
and capacity analysis, is `sglang-glm53-flash-sm120`.

## What this image is

- Base: `lmsysorg/sglang:glm-5.3-flash-amd64`, pinned by immutable digest.
- FlashInfer 0.6.18 rebuilt from source with `FLASHINFER_CUDA_ARCH_LIST=12.0f`.
  The vendor wheel ships no 12.0f cubins, and workstation Blackwell lacks
  TMEM/`tcgen05`/`wgmma`, so sm_100 and Hopper kernels do not run on it.
- Build-time gates that fail *before* the ~40-minute FlashInfer compile: base
  CUDA version, `glm5_next` importability, and the `mxfp4-pack-quantized`
  contract.

## Provenance is weaker here than in the Qwen3.8-Flash-Next build

That build pins sglang main by commit and tree, applies an archived patch, and
asserts the effective tree hash. **None of that is possible for GLM-5.3-Flash**:
`glm5_next` is not in sgl-project/sglang main as of 2026-08-27, and the vendor
per-model image is built from `ADD sglang.tar.gz` with
`SGLANG_BUILD_COMMIT=unknown`.

So the SGLang layer is pinned **by digest only**. We claim byte-level
reproducibility of the image we started from, not source-level reproducibility
of SGLang. `scripts/verify-patches.sh` asserts exactly that and fails if the
lock ever claims otherwise. When `glm5_next` lands upstream, switch to the
main+patch+tree discipline.

## Verify

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

## Scope limits

SM120 and linux/amd64 only. **Not yet built and not yet qualified on hardware.**
No performance or quality claim is made at rc.1. See `CHANGELOG.md` for the
open upstream issues that gate MTP and the native MXFP4 MoE kernel path.
