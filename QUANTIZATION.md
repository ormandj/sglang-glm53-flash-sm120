# Quantizing GLM-5.3-Flash to MXFP4A16 for two 96 GiB Blackwell cards

Written 2026-08-27, two days after the model shipped. The arithmetic and the
measured library semantics below should age well. The upstream bug list will
not — re-verify before relying on it.

---

## The problem

`zai-org/GLM-5.3-Flash` is 321.3 B total / 18 B active (`glm5_next`, MIT). The
native FP8 checkpoint is **305.81 GiB**. Two RTX PRO 6000 Blackwell cards give
**192 GiB**. It does not fit, and no 4-bit checkpoint existed — every public
release was NVFP4, and RedHat's GLM-5.2 MXFP4 had landed **63 days** after that
model shipped.

## The decision, and why it is counter-intuitive

Weights are ~90 % of the VRAM budget, so the KV pool is a small *residual* and
the weights↔context curve is violently nonlinear. The exchange rate is what
settles the design:

> **0.25 bpw off the 311.7 B expert set frees 8.9 GiB.
> Keeping ALL attention at BF16 costs 5.67 GiB.**

So moving NVFP4 (4.5 bpw) → MXFP4 (4.25 bpw) *pays for* full-precision attention
and leaves change:

| expert format | attention | weights (VRAM) | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|---|---|
| NVFP4 4.500 | ALL BF16 | 182.5 GiB | 93 k | 47 k | 23 k | 12 k |
| NVFP4 4.500 | ALL FP8 | 176.9 GiB | 520 k | 260 k | 130 k | 65 k |
| **MXFP4 4.250** | **ALL BF16** | **173.4 GiB** | **785 k** | **393 k** | **196 k** | **98 k** |
| 4.125 int4 g128 | ALL BF16 | 168.8 GiB | 1.13 M | 566 k | 283 k | 141 k |

MXFP4-with-BF16-attention beats NVFP4-with-FP8-attention on **both** context and
quality. No axis favours the NVFP4 line.

NVFP4 and MXFP4 use **identical E2M1 4-bit values**. They differ only in scale
granularity — an FP8 scale per 16 elements versus an E8M0 scale per 32. That
half-bit of block scale is the single most expensive design choice in the stack.

We did not chase 4.125 bpw int4: it abandons the E2M1 mantissa that handles
expert outliers, for context we cannot currently use.

## The recipe

`nvidia/GLM-5.2-NVFP4`'s exclusion set, with RedHat's bug fixed. **MXFP4A16**
(weight-only, group 32, E8M0 uint8 scales) on routed experts of layers 3-44 **and
the layer-45 MTP block**. Everything else BF16.

Quantize only what averages out; protect what does not:

| kept BF16 | why |
|---|---|
| ALL `self_attn`, incl. the DSA indexer | read on **every** token; 57 % of decode bandwidth |
| ALL `mlp.shared_experts` | fires on every token — its error never averages across routing |
| layers 0-2 entirely | classic early-layer sensitivity hotspot |
| `mlp.gate` (router) | a small logit perturbation changes expert **identity** discontinuously |
| mHC (`hc_*`) | error is reused across layers and 20 Sinkhorn iterations |
| DSA indexer | drives a discrete top-2048 selection; RedHat's own script flags `indexer.weights_proj` "sensitive to quantization" |
| `eh_proj` | MTP projection, on the speculative-draft path, ~0.05 GiB to protect |
| vision tower, `lm_head`, `embed_tokens` | |

Routed experts are top-8 of 288, so each expert's error is diluted by routing.
They are the only place worth spending bits — and they are 97 % of the model.

`dt_bias`, `A_log` and `{q,k,v}_conv1d` are **not** `nn.Linear`, so the KDA
recurrence gating is protected by construction under any Linear-targeted recipe.

**Exactly 37,152 tensors are quantized** = 43 × 288 × 3. Assert that number; it
is the cheapest possible guard against recipe drift, and it is what caught two
mistakes during development.

## Verified numerics

Established by **running compressed-tensors 0.18.0**, not by reading code.

- `calculate_qparams()` returns a **float32** scale already snapped to exact
  powers of two (via `generate_mx_scales`). It does **not** pre-round to uint8,
  so `MXFP4PackedCompressor._compress_scale` applying `127 + floor(log2(scale))`
  is the *only* float→E8M0 conversion. **The double-conversion hazard we
  suspected does not exist.**
- `compression_param_names()` == `("weight_packed", "weight_scale")`. No global
  scale: MXFP4 uses GROUP strategy, unlike NVFP4's TENSOR_GROUP.
- A `(2048, 4096)` expert → `weight_packed (2048, 2048) uint8` +
  `weight_scale (2048, 128) uint8` = **4.2500 bits/weight exactly**.
- **Source FP8 layout**, read from a real shard header:
  `gate_proj.weight F8_E4M3 [2048, 4096]` with
  `weight_scale_inv F32 [16, 32]` — block 128×128, all dims divide evenly.
- **End-to-end replay on synthetic FP8:** 0.1125 rel L2 against the
  FP8-dequantized reference. **Negative control** — quantizing the raw FP8
  payload *without* applying `weight_scale_inv`, the silent-corruption mode —
  measures **5366**. A verifier threshold of 0.30 separates them by orders of
  magnitude.
- safetensors container overhead is **0.002 %**, so it never explains a size
  discrepancy.

**Production run:** 37,152 quantized / 186 dequantized-to-BF16 / 1,432
passthrough; artifact 172.24 GiB across 62 shards; probe round-trip
**rel_l2_err 0.1220**.

## Why we stream over checkpoint tensors instead of loading the model

`transformers` implements `glm5_next`'s experts as **fused 3D parameters**:

```python
class Glm5NextTextExperts(nn.Module):
    gate_up_proj = nn.Parameter(empty(num_experts, 2*intermediate, hidden))
    down_proj    = nn.Parameter(empty(num_experts, hidden, intermediate))
```

So `targets="Linear"` matches **zero** expert projections, and llm-compressor's
MoE linearizer has **no `glm5_next` entry** (the registry has `glm4_moe`,
`glm4_moe_lite`, `glm_moe_dsa` — not ours).

Loading the model would also route through transformers' fine-grained-FP8
integration, where `dtype=bfloat16` does **not** guarantee the block scales are
applied. The failure mode is quantizing raw FP8 payloads — garbage with
perfectly plausible shapes.

Operating on checkpoint tensors sidesteps both. The checkpoint stores per-expert
2D tensors; we dequantize FP8 explicitly and assert it. The output layout matches
`LibertAIDAI/GLM-5.3-Flash-NVFP4`, which SGLang already loads.

## Traps

### RedHat's published GLM-5.2 recipes do not exclude layers 0-2

```python
ignore=[
    r"re:^model\.layers\.[0-2]\..*"      # <-- no comma
    r"re:.*mlp\.gate.*",
```

Python concatenates adjacent string literals into one unmatchable regex. Their
shipped configs confirm it: only 6 of 19,543 ignore entries touch layers 0-2, all
`self_attn.indexer`. **Do not copy their ignore list verbatim.**

### `device_map="auto_offload"` is not a stock Accelerate mode

It is intercepted only inside `compressed_tensors.offload.load_offloaded_model()`
(or `llmcompressor.utils.load_context()`), which patches a *specific* model
class's `from_pretrained`. Calling `from_pretrained` directly with it fails.

### A verifier that cannot fail

Our own first verifier computed "packed tensors that are not experts" and raised
if non-empty. With **zero** packed tensors that list is also empty — so a dense
BF16 no-op printed "PASSED". Assert **exact counts and config values**, never
just absence-of-bad.

### Published evals cannot resolve these choices

GPQA-Diamond is **198 questions**: at ~90 % accuracy, 1 σ = 2.13 points.
`nvidia/GLM-5.2-NVFP4` and `RedHatAI/GLM-5.2-NVFP4` disagree by **1.68 points on
the same base model** — larger than any quantization effect either reports.
Resolving a 1-point difference at 80 % power needs ~14,000 questions.

Measure **token-level distributional agreement** instead — teacher-forced top-1
agreement, KL divergence and greedy-divergence length over 200-500 k tokens of
real agentic transcripts. That is ~2,500× the sample size. And make it
**depth-resolved**: watch whether divergence *grows with position*, the signature
of error compounding through the 34 KDA recurrent layers, which is the failure
mode unique to this hybrid architecture and the one no public eval covers.

## Capacity notes that bear on the serving envelope

- **KV is 6.875 KiB/token per GPU.** Only the 11 DSA layers grow (MLA latent 512
  × 11, plus indexer 128 × 11, at fp8). `qk_rope_head_dim` is 0, so no rope cache.
- **The MLA latent is REPLICATED across TP=2** — shared across all heads, so each
  rank needs all of it. Real cost is **14,080 B/token across the pair**;
  1 GiB = 76,260 pooled tokens. Getting this wrong doubles your estimate.
  Sanity-checked against a running DSv4-Flash profile to within 3 %.
- **`--mem-fraction-static` covers weights + KV pool.** Activations and CUDA
  graphs live in the `1 − mf` remainder.
- **Recurrent state is 72.78 MiB/sequence, allocated per `--max-running-requests`,
  not per live request.** SGLang defaults the SSM dtype to **FP32** —
  `--mamba-ssm-dtype bfloat16` is mandatory, worth ~2.2 GiB at 8 slots.
- **DP-attention is not a lever** below ~4.05 bpw: it removes KV replication but
  forces the 12.4 GiB non-expert set to be replicated per rank instead of sharded.
- **EP=2 saves ≈ 0 VRAM** and costs ~27 % routing imbalance at C=1.
- **HiCache is a prefix-reuse tier, not a max-context lever** — though restores
  are cheap (128 k prefix ≈ 18 ms over PCIe Gen4).
- **Disk (172.21 GiB) and VRAM residency (173.4 GiB) are different numbers**; the
  difference is vision/router/mHC replicated across both ranks.

## Reproducing

The quantizer runs as a Kubernetes Job in the private homelab GitOps repository
(`tooling/glm53-flash/quantize-glm53-flash-mxfp4-job.yaml`). It is CPU-only —
RTN weight-only quantization is elementwise, ~6 TFLOP across the whole model, so
it is I/O-bound and never contends with the GPUs. Wall time is dominated by the
328 GB download.

Toolchain: `torch==2.13.0` (CPU), `compressed-tensors==0.18.0`,
`safetensors==0.6.2`. Pin these — the MX scale behaviour above was verified
against exactly that `compressed-tensors`.
