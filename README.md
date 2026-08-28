# SGLang GLM-5.3-Flash SM120

Immutable SGLang container source for serving an in-house MXFP4A16
quantization of
[`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
on two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120) at TP=2, with vision and the
checkpoint's native MTP/NEXTN block retained.

**`v0.1.0-rc.3` is being built and is not qualified.** No model-quality,
throughput, acceptance-rate, or maximum-context claim is made yet.
[`BENCHMARKS.md`](BENCHMARKS.md) is the evidence boundary.

## Why this quant

The official BF16 checkpoint has 642.65 billion tensor bytes and cannot fit in
the pair's measured 191.184 GiB of physical framebuffer. The deleted first
attempt requantized the official FP8 checkpoint. rc.3 instead starts from the
immutable BF16 revision and quantizes
only the routed expert projections in layers 3 through 45:

```text
43 layers x 288 experts x 3 projections = 37,152 MXFP4 tensors
```

Everything else stays BF16, including attention, DSA indexers, KDA recurrent
parameters, shared experts, routers, early layers, embeddings, LM head, mHC,
the complete 347-tensor vision path, and the MTP support projections. Layer 45's
routed experts are quantized so the native MTP block remains present without an
approximately 10 GiB BF16 draft penalty.

MXFP4 costs 4.25 bits/weight versus approximately 4.5 for NVFP4. The difference
is about 8.9 GiB across this model's expert set, material when KV is whatever
remains after roughly 173 GiB of estimated total model residency. This does not
mean MXFP4 always has lower tensor error: it is the best credible
quality/capacity/runtime choice for the current SM120 SGLang paths. The full
audit and fail-closed recipe are in [`QUANTIZATION.md`](QUANTIZATION.md).

## Runtime design

- Linux/amd64, exactly two RTX PRO 6000 Blackwell cards, TP=2.
- Vision is explicitly enabled; no language-only mode is used.
- `flashinfer_mxfp4` provides the native SM120 MXFP8-by-MXFP4 MoE path.
- Shared-expert fusion is disabled so the protected BF16 shared expert never
  enters the MXFP4 routed-expert buffer.
- FP8 E4M3 target KV and TRT-LLM DSA backends conserve memory.
- `--mamba-ssm-dtype bfloat16` is mandatory. The 34 KDA layers allocate about
  72.78 MiB of recurrent state per configured request slot at BF16.
- Expert Parallel is intentionally absent: at TP=2 it saves essentially no
  model memory and adds routing imbalance plus PCIe all-to-all traffic.
- Custom all-reduce is allowed to self-test PCIe P2P and fall back to NCCL; it
  is not forcibly disabled before measurement.

The launcher supports three A/B modes through `SPECULATIVE_MODE`:

| mode | behavior |
|---|---|
| `mtp` | default; native layer-45 NEXTN through EAGLE, adaptive 5-step / top-k 1 / 6-token profile |
| `dflash` | pinned `incoai/GLM-5.3-Flash-DFlash2`, block 8, 2,048-token draft window, FP8 draft KV |
| `none` | verifier-only baseline |

DFlash2 is a speed candidate, not a replacement chosen in advance. Current
SGLang charges its physical draft KV pool against the target token pool, so
native MTP may remain preferable when maximum pooled context matters. Both must
be measured on these cards.

## Provenance, stated honestly

`glm5_next` is not in `sgl-project/sglang` main as of 2026-08-27; model-support
PR #36507 remains open and conflicting. The vendor per-model image was built
from a tarball and reports no verifiable SGLang commit.

The base is therefore pinned by immutable OCI index and amd64 manifest digests.
This repository does not claim a vendor SGLang git revision. For the two files
we modify, rc.3 asserts exact vendor preimage SHA-256 values, applies archived
patch bytes with zero fuzz, asserts exact postimage values, and runs semantic
tests. The patches:

1. preserve GLM's contiguous gate/up layout and `(alpha=1, beta=0, limit=10)`
   SwiGLU contract in the SM120 MXFP4 loader;
2. apply upstream PR #36708's GLM DFlash2 hidden-state capture change.

FlashInfer 0.6.18 is built from exact main commit
`cbcbce48e817c83f03ad5a3e6ce59480eaf6935d` and tree
`d3a639d6f268b8bfc679a8bd15581a6a6b319a16` with
`FLASHINFER_CUDA_ARCH_LIST=12.0f`. The vendor wheel does not carry the required
SM120 cubins.

## Build and run

```bash
docker build --platform linux/amd64 \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-glm53-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.1.0-rc.3 .
```

```bash
export MODEL_DIR=/models/zai-org/GLM-5.3-Flash-BF16-MXFP4
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v2
export SPECULATIVE_MODE=mtp
./examples/serve-glm53-flash.sh
```

For the DFlash2 A/B, also set:

```bash
export SPECULATIVE_MODE=dflash
export DFLASH_DIR=/models/incoai/GLM-5.3-Flash-DFlash2
```

See [`RUN.md`](RUN.md) for diagnostics and the qualification sequence. This
repository contains no model weights and no published image.

## Verification

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

The last command needs network access. It reproduces the exact FlashInfer tree,
re-resolves the base image digests, verifies every archived patch/test byte,
and refuses to imply SGLang source provenance that the vendor image does not
provide.

## Scope

SM120 and linux/amd64 only. No stable-release, SM121, arm64, HiCache, or
production-readiness claim. A successful image build makes rc.3 built; only
exact-candidate evidence can make it qualified.
