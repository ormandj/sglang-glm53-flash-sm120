# GLM-5.3-Flash BF16 to MXFP4 quantization

This is the quantization contract for `v0.1.0-rc.5`. The production artifact is
still being built and is not qualified. Numbers described as estimates are not
hardware measurements.

## Decision

Start from the official BF16 checkpoint, not the official FP8 checkpoint:

- repository: `zai-org/GLM-5.3-Flash-BF16`
- immutable revision: `f12e0fe1f6b2ea274c11a569582edfd99d993c5e`
- 120 safetensors shards
- 642,646,653,816 indexed tensor bytes

Quantize only the routed expert projections in language-model layers 3 through
45 with compressed-tensors `MXFP4A16`. Layer 45 is the checkpoint's native MTP
block and is deliberately retained. Everything else remains BF16, including the
entire vision tower.

The exact target is 37,152 tensors:

```text
43 layers x 288 experts x 3 projections = 37,152
```

The projections are `gate_proj`, `up_proj`, and `down_proj`. The quantizer uses
an anchored regex over the full checkpoint name and fails if the resulting
namespace differs by even one tensor.

## Why the FP8-derived artifact was rejected

The deleted artifact was structurally loadable, but it performed FP8 to MXFP4
requantization. That is unnecessary compound quantization now that the official
BF16 source exists.

Checks against corresponding BF16 and FP8 tensors found:

- ordinary routed experts differ by roughly 0.0017 relative L2 before MXFP4;
- the layer-45 MTP expert sample differs by roughly 0.021, indicating that the
  two published checkpoints are not interchangeable at the draft layer;
- on sampled expert projections, direct BF16 to MXFP4 error was approximately
  0.109 to 0.122 relative L2;
- using FP8 as the source increased that MXFP4 error by 2.35% on average and up
  to 7.17% in the sampled set.

These are tensor reconstruction checks, not model-quality scores. They are
sufficient to reject the avoidable FP8 intermediate, but not to qualify the
finished model.

The old FP8 source, old output, and old serving copy were removed from the
scratch and model volumes before the new run began.

## Why MXFP4, not NVFP4

This is a system-level choice, not a claim that MXFP4 always has lower weight
error.

MXFP4 stores E2M1 values with one E8M0 scale per 32 weights, for exactly 4.25
bits/weight. NVFP4 uses finer scale granularity and costs approximately 4.5
bits/weight. Across this model's routed experts, that quarter-bit difference is
about 8.9 GiB of total residency. On the measured 191.184 GiB framebuffer pair,
those bytes are a large fraction of the memory left for KV after weights.

The current SGLang paths also matter:

- the SM120 `flashinfer_mxfp4` path uses MXFP8 activations with MXFP4 weights;
- the native NVFP4 MoE path is effectively an FP4-activation path;
- a per-layer NVFP4/MXFP4 mixture is not currently routed correctly by SGLang's
  quantization-scheme selection, and the latest model-free fusion matcher does
  not make that hybrid checkpoint safe.

For this hardware and software stack, routed-expert MXFP4 plus BF16 protected
components gives the best credible quality/capacity starting point. NVFP4
remains a valid future A/B only if its end-to-end runtime, model quality, and KV
capacity are measured on the same cards.

An experimental per-group exponent MSE search improved sampled reconstruction
error by only about 1.06% over the standard MXFP4 max-absolute scale rule. The
result would still be a valid E8M0/MXFP4 encoding, but it would replace the
audited ecosystem producer with a custom scale optimizer. Unweighted weight MSE
also does not establish lower activation error or better model quality, so that
small tensor-level result is not enough to justify the custom producer risk.

## Protected BF16 scope

The following are copied bit-for-bit from the BF16 source:

| component | reason |
|---|---|
| vision tower and multimodal projection | vision is required and must not be silently degraded |
| embeddings and LM head | high reuse and direct effect on token probabilities |
| all attention and DSA indexer tensors | active on every token; indexer affects discrete sparse selection |
| all KDA recurrent-state parameters | error can compound with sequence depth |
| mHC parameters | reused across layers |
| shared experts | active on every routed token |
| routers | small changes can alter expert identity |
| language-model layers 0 through 2 | conservative early-layer protection |
| MTP `eh_proj` and non-expert tensors | small relative to the expert block and draft-sensitive |

There are 1,618 protected tensors in the source namespace, including exactly
347 vision tensors. The production verifier compares every protected tensor's
dtype, shape, and bytes.

## On-disk format and runtime contract

The output uses compressed-tensors `mxfp4-pack-quantized`:

- weight type: 4-bit float E2M1
- strategy: `GROUP`
- group size: 32
- activation quantization in the checkpoint: none
- packed parameter: `weight_packed`, `uint8`
- scale parameter: `weight_scale`, `uint8` E8M0

A `(2048, 4096)` projection becomes a `(2048, 2048)` packed tensor plus a
`(2048, 128)` scale tensor, exactly 4.25 bits/weight.

The serialized config also carries `ignore: ["re:.*"]` as an SGLang runtime
contract. SGLang selects MXFP4 for `FusedMoE` from the global format before it
checks ignores, while ordinary unmatched `LinearBase` modules otherwise fail
compressed-tensors target resolution. The catch-all therefore leaves every
ordinary linear on its stored BF16 path without disabling the routed-expert
MXFP4 method. Serving must set `--disable-shared-experts-fusion`; otherwise the
protected BF16 shared expert is appended to the MXFP4 fused expert buffer.

SGLang's inherited SM120 post-loader was GPT-OSS-specific: it assumed pairwise
gate/up rows and hard-coded SwiGLU-OAI `(alpha=1.702, beta=1, limit=7)`. GLM's
per-expert loader produces contiguous `[gate; up]` halves and GLM uses standard
clamped SwiGLU `(alpha=1, beta=0, limit=10)`. v0.1.0-rc.5 applies a byte-gated patch
that preserves both contracts and has a build-time semantic test.

## Fail-closed production procedure

[`quantization/quantize_glm53_bf16_mxfp4.py`](quantization/quantize_glm53_bf16_mxfp4.py)
performs model-free, streaming quantization. Loading the Transformers model is
both unnecessary and risky because its experts are fused 3D parameters while
the checkpoint exposes the per-expert 2D tensors that SGLang loads.

Before writing, the script verifies:

- source repository revision, file count, byte count, and completion marker;
- exact BF16 config, multimodal config, processor files, and tensor namespace;
- exact target count, shapes, and BF16 source dtypes;
- absence of any source quantization config.

It writes to an `.incomplete` sibling and refuses to overwrite either an old
temporary output or a final output. After conversion it verifies:

- the exact serialized quantization config, including the runtime ignore;
- 37,152 packed tensors and 37,152 scale tensors with exact uint8 shapes;
- all 1,618 protected tensors bit-for-bit;
- all eight ancillary tokenizer, processor, template, license, README, and
  repository-metadata files byte-for-byte;
- one dequantization probe for each projection in every targeted layer;
- bounded total tensor bytes;
- SHA-256 for every artifact file and full installed-package provenance.

Only after every check succeeds does it write `.quant-complete`, sync the
filesystem, and atomically rename the directory into its final path.

Pinned toolchain for this run:

- PyTorch `2.13.0+cpu`
- compressed-tensors commit `aa91ea52e9cb44da4f984dd53b4c2df65ef554b4`
  (tree `5071ae29e82a01663585bef999923fe424bdf236`)
- llm-compressor commit `d1e1fb6cb2ad2c99563164be36c2f83d846462b4`
  (tree `23a003ddb2215804851b41cbc0844d428e207f28`)

The exact commits were installed from verified git trees. A synthetic
GLM-shaped checkpoint was serialized and loaded successfully before the
production job was launched; its reconstruction check was 0.112915 relative
L2. That result validates the toolchain and format only.

## Capacity implications

The artifact's exact tensor payload is 184,905,481,080 bytes, or 172.207 GiB.
The pre-run residency estimate is about 173.4 GiB across TP=2 after accounting
for TP placement and replicated tensors. The cards expose 95.592 GiB each
(191.184 GiB total); `--mem-fraction-static 0.96` therefore caps weights plus KV
at 183.536 GiB total. These estimates must be replaced by measured GPU residency
after the exact candidate boots.

At FP8 KV, the target DSA cache is estimated at 6.875 KiB/token/GPU. A 524,288-
token pool therefore consumes 3.438 GiB/GPU. The 34 KDA layers also allocate
recurrent state per request slot, not per active request;
`--mamba-ssm-dtype bfloat16` keeps that to about 72.78 MiB/slot, or 0.569
GiB/GPU at eight slots. After estimated residency, KV, and recurrent state, the
static budget has only about 1.06 GiB/GPU remaining. This is why the launcher
uses FP8 KV, eight slots, and a 524,288-token context ceiling, and why startup
capacity is a hard qualification gate.

Native MTP reuses the checkpoint's quantized layer 45 and is the capacity
baseline. DFlash2 adds a separate draft cache. The launcher therefore pins its
draft KV to `fp8_e4m3` and its logical window to 2,048 tokens, but current
SGLang still sizes the physical draft pool against the target token pool. The
achievable pooled context for both modes must be read from startup logs and
tested, not inferred from the configured context length.

## Qualification still required

The completed artifact is not accepted on structure alone. Qualification must
cover:

- target-only, native MTP, and DFlash2 load/boot on TP=2;
- vision input and image-dependent output;
- token-level agreement or KL against the BF16 teacher on representative text
  and multimodal traces;
- long-context divergence through the KDA recurrence;
- output throughput, latency, speculative acceptance, and actual KV pool;
- cold prefill and first decode above 262k tokens because of open SGLang issue
  #36550.

Performance and quality results belong in `BENCHMARKS.md` with evidence from
this exact immutable candidate.
