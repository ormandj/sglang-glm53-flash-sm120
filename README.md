# GLM-5.3-Flash on dual SM120

This repository builds one immutable SGLang candidate for high-quality
GLM-5.3-Flash inference on two RTX PRO 6000 Blackwell Server Edition GPUs.
The intended serving profile is TP=2, EP=1, vision enabled, native adaptive
MTP, FP8 KV, a roughly 500K shared token pool, and practical C4 agentic fanout.

Current image: `sglang-glm53-flash-sm120:v0.1.0-rc.18`.

**v0.1.0-rc.18 is built, but it is not a qualified release.** Its immutable
image digest and import identities passed. Model load, coherent output, quality,
context capacity, MTP acceptance, vision, and performance remain separate gates
recorded in [BENCHMARKS.md](BENCHMARKS.md).

## What changed in v0.1.0-rc.18

The rc.18 source and runtime trees are identical to rc.17. It additionally
fetches and verifies ModelOpt's exact `0.47.0rc0` tag at the pinned commit so
`setuptools-scm` produces the locked package version during the immutable
container build. The cache schema remains `v10` because no runtime code changed.

The failed in-house MXFP4 artifact and all of its runtime diagnostic patches
were removed from the active build. Its persistent evidence remains so the
failure cannot be accidentally reinterpreted as a serving bug.

The candidate now uses exact source trees:

- SGLang integration `42a56dc505f775d6f54e9d27a9b57c66023420a0`,
  based on current upstream main, with GLM-5.3 support, raw-layout FP8 TileLang
  DSA, and an explicit E4M3-K32 W4A16 loader/runner contract.
- FlashInfer `008122fa75c7a27c839feea57a6ef8e8846fa265`, containing
  upstream's large W4A16 expert-bank fix and the matching SM120 weight
  preparation contract.
- NVIDIA ModelOpt `022767c7ab3d7d36211affd85e5c496770cde768`, used for
  the controlled MSE-calibrated quantization recipe.

The vendor image is pinned by its linux/amd64 manifest and supplies only the
known CUDA/PyTorch environment. Its unverifiable SGLang Python tree is shadowed
by the exact integration checkout. [stack.lock.json](stack.lock.json) records
the repositories, commits, trees, base manifests, and intended runtime profile.

## Quantization direction

The selected local recipe is routed-expert W4A16: E2M1 FP4 weights with E4M3
per-32-element K-group scales, one FP32 global scale, and BF16 activations. It
preserves vision, embeddings, LM head, routers, shared experts, and initially
all attention/KDA/mHC linears in BF16. Native layer-45 MTP routed experts use
the same measured recipe so keeping MTP does not add an unbounded BF16 expert
bank. See [QUANTIZATION.md](QUANTIZATION.md) for the contract and quality gates.

This is not the public g16 NVFP4 layout and not the rejected E8M0 MXFP4 layout.
The g32 scale overhead is 0.25 bits per weight, while fractional E4M3 scales
avoid MXFP4's power-of-two-only block scaling. G32 is coarser than the public
g16 control and therefore is not inherently higher quality; it is the measured
size tradeoff needed for this pair. Whether the pinned MSE recipe meets the
quality target will be decided against the BF16 teacher, not assumed from
format names.

## Runtime target

The default launcher requests:

- two GPUs with TP=2 and no expert parallelism;
- vision and the `glm45` reasoning / `glm47` tool parsers;
- native adaptive EAGLE/NextN MTP at 5 steps, top-k 1, 6 draft tokens;
- raw-layout FP8 TileLang DSA and FP8 E4M3 KV;
- separately executed BF16 shared experts, because fusing them into the
  serialized FP4 routed bank would violate the checkpoint contract;
- 524,288 total cache tokens, C4, 20 BF16 recurrent-state slots, and decode
  CUDA graphs through batch size 4.

The 20 recurrent slots are legitimate GLM hybrid-model state: the runtime uses
five physical KDA/Mamba slots per live request. They are separate from the KV
token pool. C8 will be tested with 40 slots after C4 correctness and capacity.
HiCache will be evaluated later for reusable prefixes; it does not turn inactive
host-resident blocks into active GPU token capacity.

## Build and verify

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh

docker build \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-glm53-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.1.0-rc.18 .
```

The last verifier uses the network to reproduce all three pinned source trees
and re-resolve the immutable base manifests. Build output alone must not be
described as qualified.

## Run

Follow [RUN.md](RUN.md) or use [examples/serve-glm53-flash.sh](examples/serve-glm53-flash.sh).
The launcher defaults are the desired qualification envelope, not yet a memory
or throughput claim.
