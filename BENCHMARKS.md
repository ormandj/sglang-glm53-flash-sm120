# Measured results

Two measurement sets are recorded here. The v0.1.1 set was measured on
2026-09-02 on the released `v0.1.1` image (internal digest
`sha256:5e499c5f...`), one host: 2x NVIDIA RTX PRO 6000 Blackwell (96 GB,
SM120), TP2, PCIe, no NVLink, serving
`GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO` with the configuration in
[`RUN.md`](RUN.md) (pool and context 499,712 tokens, chunked prefill 4,096,
one 4,096-token chunk per extend batch, four running requests, mamba pool
28, native EAGLE MTP 5/1/6 adaptive, FP8 E4M3 KV, HiCache 32 GB, KDA
extend block 2,048, CPU image preprocessing). The v0.1.0 set (2026-08-31,
pool 507,904, chunk 2,048, mamba 20, HiCache 128 GB) is kept where v0.1.1
did not re-measure a row and is marked as such. Raw receipts live in
[`evidence/`](evidence/); the harness is in [`bench/`](bench/).

## Capacity (v0.1.1)

| Probe | Result |
|---|---|
| KV/token pool | 499,712 tokens (equals the declared context limit) |
| Largest single prefill served | 494,592-token prompt; pool usage peaked at 495,616 tokens (99.2%) |
| Cold single-request prefill | 204,800 tokens in 34.1 s TTFT, 6,000 prompt tok/s |
| 358,400 and 494,592-token prompts | served (they extend the 200k prompt's prefix, so 153k and 136k new tokens on a cached prefix; not a cold rate) |
| C4 fill | 4 x 122,880-token prompts concurrently, TTFT 2.4 s on cached prefixes |
| Sustained C4 | 8/8 distinct-prefix waves (4 x 29,551 tokens each), 255 to 269 aggregate output tok/s, zero restarts, zero errors |
| Device headroom at 99% pool usage | effectively zero: four recoverable allocator segment-mapping warnings (`expandable_segments: memory mapping failed with OOM ... 2097152 bytes`, 1.7 MB free) during the 350k and 494k prefills and the GSM8K start; the allocator released cached blocks and every request completed, no `OutOfMemoryError`, no restart |
| Peak mamba pool usage | 0.93 of 28 slots |
| HiCache host tier | 2,648,320 KV tokens (20.85 GB, packed MTP layers) + 11.16 GB mamba tier (page-first) at the 32 GB setting |

## Decode (v0.1.1, shipped configuration)

Engine gate `glm-qualification` (bench/aiperf, pinned aiperf 0.12.0 at
6ed4823d): cohort-only cells (requests equal concurrency), coding corpus,
4,096 output tokens, temperature 0, five repetitions per cell, every shape
warmed once before the timed cells, run inside the serving pod on a warm
server. Each repetition is the OLS rate of the server's decode-token and
forward-pass counters inside the analyzer's plateau window; every listed
repetition is analyzer-valid with zero flags; average decode context 18.6k
to 19.0k tokens. Forward passes per second is the engine step rate; tokens
per second is after MTP acceptance.

| Cell | OLS tok/s per repetition | Mean tok/s | Median tok/s | Forwards/s per repetition | Mean fwd/s | Accepted tok/fwd/req (median) |
|---|---|---:|---:|---|---:|---:|
| C1 | 150.9 / 204.2 / 175.1 / 155.5 / 186.2 | 174.4 | 175.1 | 56.1 / 49.5 / 51.5 / 55.2 / 51.0 | 52.7 | 3.5 |
| C2 | 282.0 / 272.3 / 229.8 / 231.7 / 244.5 | 252.1 | 244.5 | 34.2 / 38.4 / 40.6 / 39.6 / 39.1 | 38.4 | 3.2 |
| C3 | 296.7 / 297.7 / 310.4 / 281.4 / 308.0 | 298.8 | 297.7 | 34.2 / 32.5 / 32.2 / 34.3 / 31.9 | 33.0 | 3.0 |

v0.1.0 on the same harness: C1 169.7 tok/s at 54.0 fwd/s, C2 235.7 at
40.4. The engine step rate is unchanged within repetition spread; the
higher token rates come from acceptance.

C4 is not a qualified cell on v0.1.1. The refill-style `repeat-c4` cell
(12 requests, the shape v0.1.0 reported as "C4 sustained") is analyzer-
rejected on all five repetitions: refill prefills land inside the plateau
and exact occupancy holds for 95.8% of the window. A true four-request
cohort never decodes four at once: the server's decode concurrency peaks at
3, each decoding request holds 4 mamba slots (12 used), 13 to 21 slots are
evictable radix-cached states and only 2 are free of 28, and admission does
not evict cached mamba states to admit the fourth. The fourth request waits
in prefill or the queue. Numbers from rejected cells are not quoted; the
C4 wave receipts above are the sustained four-request evidence.

MTP acceptance on the coding corpus is 2.7 to 4.2 accepted tokens per
forward per request. The math-content C1 figure (257 to 267 tok/s at
acceptance about 6) is a v0.1.0 measurement and was not repeated.

## Prefill (v0.1.1)

Engine gate cold-prefill panel: five cold, cache-busted requests per
length at C1, one output token, temperature 0. Rate is aggregate prompt
tokens per second over the cell window.

| Shape | Rate | Median TTFT |
|---|---:|---:|
| 8k (8,205 tokens) | 5,263 tok/s | 1.55 s |
| 32k (32,781 tokens) | 5,903 tok/s | 5.55 s |
| 64k (65,549 tokens) | 5,918 tok/s | 11.1 s |
| 128k (130,829 tokens) | 5,918 tok/s | 22.2 s |
| 200k (204,800 tokens, ladder, C1 cold) | 6,000 tok/s | 34.1 s |

v0.1.0 measured 5,153 tok/s at 200k cold and 4,985 tok/s aggregate on
4 x 120k concurrent cold; the latter shape was not repeated cold on
v0.1.1 (the ladder's fill ran on cached prefixes).

## Quality (v0.1.1)

GSM8K, all 1,319 test questions, zero-shot, temperature 0, seed 42,
16,384-token cap, concurrency 4, served through the OpenAI-compatible API
with reasoning enabled (bench/aiperf `configs/gsm8k.yaml`).

| Grader | Correct | Accuracy |
|---|---:|---:|
| Pinned AIPerf grader (last-number fallback) | 1,174 / 1,319 | 89.0% |
| Position-based regrader ([`bench/regrade_gsm8k.py`](bench/regrade_gsm8k.py)) | 1,278 / 1,319 | 96.9% |

The pinned grader under-scores marker-less models: GLM-5.3 emits no `####`
answer marker zero-shot, and all 104 grader disagreements were correct
answers the fallback mis-extracted (zero downward flips). v0.1.0 scored
1,180 / 1,282 (89.5% / 97.2%) and its BF16-attention predecessor 89.0% /
97.0% with the same method.

Vision: 3840x2160 and 7680x4320 images at the model's 8,000-token budget
(7,994 prompt tokens) answer in 3.4 s and 3.0 s through the
OpenAI-compatible image input; the v0.1.0 bar-chart read-back check was
not repeated.

## Artifact reconstruction quality (producer receipts, unchanged)

| Tier | Metric | Value |
|---|---|---:|
| Routed experts (W4A16 NVFP4, K32 MSE) | aggregate relative L2 | 0.0851 |
| Routed experts | minimum matrix cosine | 0.9949 |
| FP8 [128,128]-block attention + shared experts (317 tensors) | aggregate relative L2 | 0.0225 |
| FP8 tier | minimum matrix cosine | 0.9996 |

## Known measurement gaps

- No qualified C4 decode cell (see Decode). Raising the mamba pool to
  about 36 slots would cost about 300 MB per GPU that the 0.99 memory
  fraction does not have at chunk 4,096; the fix belongs in the scheduler's
  admission path (evict cached mamba states on demand).
- The 350k and 494k ladder rungs and the C4 fill ran on cached prefixes
  because of rung ordering; only the 200k rung and the engine-gate panel
  are cold prefill rates.
- Device headroom at full pool usage is effectively zero (see Capacity);
  a workload that allocates a large transient at that point would fail.
- One quantization configuration lesson is load-bearing enough to repeat:
  quantizing the MTP draft layer (layer 45) silently collapses speculative
  acceptance to zero while leaving outputs correct. The shipped producer
  keeps it BF16. See `CHANGELOG.md` for the full narrative.
