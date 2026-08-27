# SGLang GLM-5.3-Flash SM120

Reproducible SGLang container source for serving an **in-house MXFP4A16
quantization** of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
(321 B total / 18 B active, `glm5_next`, MIT) on two NVIDIA RTX PRO 6000
Blackwell GPUs (SM120) at TP=2.

**This is `v0.1.0-rc.1`: built, not qualified.** No performance or quality claim
is made. `BENCHMARKS.md` records what has and has not been measured.

## Why quantize at all, and why MXFP4

The native FP8 checkpoint is 305.81 GiB against 192 GiB of VRAM. It does not
fit, and every public 4-bit release was NVFP4.

Weights are ~90 % of the budget, so the KV pool is a small residual and the
weights↔context curve is nonlinear. The exchange rate decides the design:
**0.25 bpw off the 311.7 B expert set frees 8.9 GiB, while keeping ALL attention
at BF16 costs only 5.67 GiB.** So NVFP4 (4.5 bpw) → MXFP4 (4.25 bpw) *pays for*
full-precision attention:

| expert format | attention | weights | C=1 | C=4 |
|---|---|---|---|---|
| NVFP4 4.500 | ALL BF16 | 182.5 GiB | 93 k | 23 k |
| NVFP4 4.500 | ALL FP8 | 176.9 GiB | 520 k | 130 k |
| **MXFP4 4.250** | **ALL BF16** | **173.4 GiB** | **785 k** | **196 k** |

MXFP4-with-BF16-attention beats NVFP4-with-FP8-attention on **both** axes. Both
formats use identical E2M1 4-bit values; they differ only in scale granularity.

Full derivation, the per-component protect/quantize rationale, the verified MX
scale semantics, and the traps we hit are in [`QUANTIZATION.md`](QUANTIZATION.md).

## Tested configuration

- Linux/amd64, 2× RTX PRO 6000 Blackwell (SM120), TP=2. **No Expert Parallel** —
  at TP=2 it saves ≈0 VRAM and adds ~27 % routing imbalance at C=1.
- `fp8_e4m3` KV, TileLang DSA backends, `--mem-fraction-static 0.96`.
- `--mamba-ssm-dtype bfloat16` is **mandatory**: SGLang defaults the SSM state to
  FP32, and the 34 KDA layers hold 72.78 MiB of recurrent state per slot,
  allocated per `--max-running-requests` rather than per live request.
- MTP is **off** at rc.1 — sgl-project/sglang#36653 and #36599 both block NEXTN
  for TP>1 with an FP4 draft.

## Provenance, stated honestly

`glm5_next` is **not** in sgl-project/sglang main (PR #36507 open as of
2026-08-27), and the vendor per-model base image is built from `ADD
sglang.tar.gz` reporting `SGLANG_BUILD_COMMIT=unknown`.

So the SGLang layer is pinned **by immutable digest only**. We claim byte-level
reproducibility of the image we started from, not source-level reproducibility of
SGLang. `scripts/verify-patches.sh` asserts exactly that and **fails** if the
lock is ever edited to imply more. When #36507 merges, switch to pinning main by
commit and tree with an archived patch.

FlashInfer 0.6.18 *is* built from a verified source tree with
`FLASHINFER_CUDA_ARCH_LIST=12.0f`; the vendor wheel carries no 12.0f cubins, and
workstation Blackwell lacks TMEM/`tcgen05`/`wgmma` so sm_100 and Hopper kernels
do not run on it.

## Build and run

```bash
docker build --platform linux/amd64 \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-glm53-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.1.0-rc.1 .
```

```bash
export MODEL_DIR=/models/zai-org/GLM-5.3-Flash-MXFP4
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v1
./examples/serve-glm53-flash.sh
```

See [`RUN.md`](RUN.md). This repository contains no model weights and no
published image.

## Reproducibility

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

The last re-fetches the pinned FlashInfer objects, reproduces its tree, and
re-resolves the base image digests against the registry.

## Scope limits

SM120 and linux/amd64 only. No SM121, arm64, NVFP4, or HiCache claim.
**Known gap:** MXFP4 MoE has no native SM120 CUTLASS path — FlashInfer #2847 and
vLLM #31085 leave three runtime guards filtering SM120, so expect the Marlin
fallback (~28 % slower prefill measured on gpt-oss-120b).
