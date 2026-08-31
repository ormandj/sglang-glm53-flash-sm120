# Measured results — v0.1.0-rc.65 + MIX-V2 artifact

All numbers below were measured on 2026-08-31 on one host: 2x NVIDIA RTX PRO
6000 Blackwell Max-Q Workstation Edition (96 GB, SM120), TP2, PCIe, no NVLink,
serving the MIXED_PRECISION artifact described in
[`QUANTIZATION.md`](QUANTIZATION.md) with the exact configuration in
[`RUN.md`](RUN.md) (pool and context 507,904 tokens, chunked prefill 2,048,
C4, native EAGLE MTP 5/1/6, FP8 E4M3 KV, HiCache 128 GB). Raw receipts live
in [`evidence/`](evidence/); the harness that produced them is in
[`bench/`](bench/).

## Capacity

| Probe | Result |
|---|---|
| KV/token pool | 507,904 tokens (equals the declared context limit) |
| Cold single-request prefill | 200k, 350k, and 502,784-token prompts served, correct retrieval |
| C4 fill | 4 x 122,880-token distinct prompts concurrently |
| Sustained C4 | 8/8 distinct-prefix waves, zero restarts, zero errors |
| Post-capture device headroom | 2.73 GB per GPU; zero OOMs across the whole sweep |
| HiCache host tier | 13,499,968 KV tokens (82.9 GB) + mamba tier (page-first) |

## Decode (shipped configuration)

Decode is the OLS rate of the server's decode-token counter inside the
analyzer's plateau window — prefill-free by construction, shapes warmed
before any timed cell (coding corpus, 4,096 output tokens, temperature 0,
n=5). C1/C2 are pure equal-context plateaus (zero analyzer flags). C4 uses
refill cells (requests = 3x concurrency): a 4-request cell's
exact-occupancy overlap is structurally shorter than the analyzer minimum
on a long-prefill model, so the sustained window carries four disclosed
flags (interleaved refill prefill, ~95% occupancy, context wander).

| Cell | Repetitions (OLS tok/s) | Mean | Forwards/s | Accepted tok/fwd/req |
|---|---|---:|---:|---:|
| C1 | 157.7 / 162.5 / 164.9 / 173.5 / 189.8 | 169.7 | 54.0 | 3.2 |
| C2 | 220.7 / 230.3 / 240.7 / 243.1 / 243.9 | 235.7 | 40.4 | 3.0 |
| C4 sustained | 340.3 / 343.1 / 344.9 / 349.7 / 357.2 | 347.0 | 29.5 | 3.0 |

C1-to-C4 scaling is 2.04x, matching the DeepSeek-V4-Flash publication's
2.02x on the same hardware.

MTP acceptance is ~2.5 accepted length on general content and 5.8–6.0 on
math, where the server sustains 257–267 tok/s at C1. The shipped
configuration uses the adaptive speculative ladder (candidate steps [3,5],
measured tier cost 0.39 GB), which wins on low-acceptance agentic content,
plus PCIe IPC allreduce, which also lifts cold 200k prefill from 4,599 to
5,153 tok/s. Alternatives measured and not shipped (NCCL P2P disabled,
DFlash-2 block-diffusion drafting, static speculation) have receipts under
[`evidence/`](evidence/).

## Prefill

Rates are prompt tokens divided by time to first token on cold, cache-busted
requests (ladder receipts).

| Shape | Rate |
|---|---|
| 200k cold C1 | 4,599 tok/s |
| 4 x 120k concurrent (aggregate) | 4,985 tok/s |
| 350k with warm prefix (HiCache) | 10,617 tok/s |
| 502,784 with warm prefix (HiCache) | 13,413 tok/s |

## Quality

GSM8K, all 1,319 test questions, zero-shot, temperature 0, seed 42, served
through the OpenAI-compatible API with reasoning enabled.

| Grader | Correct | Accuracy |
|---|---:|---:|
| Pinned AIPerf grader (last-number fallback) | 1,180 / 1,319 | 89.5% |
| Position-based regrader ([`bench/regrade_gsm8k.py`](bench/regrade_gsm8k.py)) | 1,282 / 1,319 | 97.2% |

The pinned grader under-scores marker-less models: GLM-5.3 emits no `####`
answer marker zero-shot, and every one of the 102 grader disagreements was a
correct answer the fallback mis-extracted (zero downward flips). The
BF16-attention predecessor artifact scored 89.0% / 97.0% with the same
method, so the FP8 tier cost no measurable accuracy.

Vision: a generated ground-truth bar chart is read exactly (title and all
four values) through the OpenAI-compatible image input path.

## Artifact reconstruction quality (producer receipts)

| Tier | Metric | Value |
|---|---|---:|
| Routed experts (W4A16 NVFP4, K32 MSE) | aggregate relative L2 | 0.0851 |
| Routed experts | minimum matrix cosine | 0.9949 |
| FP8 [128,128]-block attention + shared experts (317 tensors) | aggregate relative L2 | 0.0225 |
| FP8 tier | minimum matrix cosine | 0.9996 |

## Known measurement gaps

- The C4 decode timing cell was rejected by the strict equal-context window
  analyzer under HiCache write-back; C4 *throughput* is instead evidenced by
  the sustained-wave receipts (~213–250 tok/s aggregate on 4 x 29.5k-token
  prompts in earlier same-stack runs).
- One quantization configuration lesson is load-bearing enough to repeat:
  quantizing the MTP draft layer (layer 45) silently collapses speculative
  acceptance to zero while leaving outputs correct. The shipped producer
  keeps it BF16. See `CHANGELOG.md` for the full narrative.
