# Benchmarks

**Nothing has been measured yet.** `v0.1.0-rc.1` is built, not qualified. This
file exists so that absence is explicit rather than implied.

## What has been verified

Quantization correctness only — not model quality, not throughput:

| check | result |
|---|---|
| quantized tensors | **37,152** exactly (43 layers × 288 experts × 3 projections) |
| format | `mxfp4-pack-quantized`, `compressed-tensors` |
| scheme | 4-bit float, GROUP, `group_size` 32, `scale_dtype uint8`, `input_activations: null` |
| bits/weight | **4.2500** exactly, measured on a `(2048, 4096)` projection |
| protected & unquantized | mHC 45 · `dt_bias` 34 · `A_log` 34 · conv1d 102 · `o_proj` 46 · DSA indexer 84 · shared expert 129 · dense layer 0 · router 43 · vision tower 347 · `eh_proj` · `lm_head` |
| no source FP8 leakage | no surviving `weight_scale_inv` |
| numerical round-trip vs FP8 source | **rel_l2_err 0.1220** (threshold 0.30; negative control omitting `weight_scale_inv` measures 5366) |
| artifact size | 172.24 GiB, 62 shards |

## What has NOT been measured

- Any quality benchmark. No GSM8K, no coding benchmark, no long-context eval.
- Any throughput or latency number, at any concurrency.
- Multimodal behaviour. This is the first natively-multimodal GLM-5 model and the
  vision path is the least-tested part of any new SM120 build.
- Long-context behaviour, and specifically whether quant-vs-reference divergence
  **grows with position** — the signature of error compounding through the 34 KDA
  recurrent layers.
- Anything about MTP, which is disabled at rc.1 by two open upstream bugs.

## How quality should be measured when it is

Not with small benchmark suites. GPQA-Diamond is 198 questions: 1 σ = 2.13 points
at ~90 % accuracy, and the two published GLM-5.2 NVFP4 evaluations disagree by
1.68 points **on the same base model** — more than either claims for
quantization.

Use teacher-forced **token-level distributional comparison** against the FP8
master over 200-500 k tokens of real agentic transcripts: top-1 agreement, KL
divergence, greedy-divergence length. That is ~2,500× the sample size of a GPQA
run and resolves sub-0.1 % effects. Make it depth-resolved.

Then a head-to-head against the incumbent on the actual workload, at C={1,4,8}.
