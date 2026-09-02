# Measured results

Two measurement sets are recorded here. The v0.1.1 set was measured on
2026-09-02 on the released `v0.1.1` image (internal digest
`sha256:5e499c5f...`), one host: 2x NVIDIA RTX PRO 6000 Blackwell (96 GB,
SM120), TP2, PCIe, no NVLink, serving
`GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO` with the configuration in
[`RUN.md`](RUN.md) (pool and context 499,712 tokens on v0.1.1 and v0.1.2,
450,560 on v0.1.3, chunked prefill 4,096,
one 4,096-token chunk per extend batch, four running requests, mamba pool
28, native EAGLE MTP 5/1/6 adaptive, FP8 E4M3 KV, HiCache 32 GB, KDA
extend block 2,048, CPU image preprocessing). The v0.1.0 set (2026-08-31,
pool 507,904, chunk 2,048, mamba 20, HiCache 128 GB) is kept where v0.1.1
did not re-measure a row and is marked as such. Raw receipts live in
[`evidence/`](evidence/); the harness is in [`bench/`](bench/).

## Capacity (v0.1.1)

| Probe | Result |
|---|---|
| KV/token pool | 499,712 tokens on v0.1.1 and v0.1.2; 450,560 on v0.1.3 (equals the declared context limit) |
| Largest single prefill served | 494,592-token prompt; pool usage peaked at 495,616 tokens (99.2%) |
| Cold single-request prefill | 204,800 tokens in 34.1 s TTFT, 6,000 prompt tok/s |
| 358,400 and 494,592-token prompts | served (they extend the 200k prompt's prefix, so 153k and 136k new tokens on a cached prefix; not a cold rate) |
| C4 fill | 4 x 122,880-token prompts concurrently, TTFT 2.4 s on cached prefixes |
| Sustained C4 | 8/8 distinct-prefix waves (4 x 29,551 tokens each), 255 to 269 aggregate output tok/s, zero restarts, zero errors |
| Device headroom | see "Memory headroom" below; the v0.1.1 ladder logged four recoverable `expandable_segments` mapping warnings at 99% usage |
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

C4 was measured on `v0.1.2` (rc.1, internal digest `sha256:9f65221e...`),
which differs from v0.1.1 only by the scheduler admission fix; the same
`repeat-c4` gate run as a four-request cohort, five repetitions, all
analyzer-valid with zero flags:

| Cell | OLS tok/s per repetition | Mean tok/s | Median tok/s | Forwards/s per repetition | Mean fwd/s | Accepted tok/fwd/req (median) |
|---|---|---:|---:|---|---:|---:|
| C4 (v0.1.2) | 333.8 / 411.3 / 342.0 / 363.9 / 338.1 | 357.8 | 342.0 | 30.8 / 25.6 / 30.0 / 29.6 / 31.1 | 29.4 | 2.9 |

On v0.1.1 a four-request cohort never decoded four at once: the scheduler's
`batch_is_full` latch, set by one transient admission refusal, is cleared
only when a request finishes, so the fourth request waited for the whole
decode of the other three (three decoding, one queued). v0.1.2 re-evaluates
admission every round on hybrid SSM caches. Receipts:
`evidence/v0.1.2-rc.1-c4-admission-20260902.txt` and
`evidence/v0.1.2-engine-gates/`.

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
with reasoning enabled (bench/aiperf `configs/gsm8k.yaml`), graded with the
position-based extractor ([`bench/regrade_gsm8k.py`](bench/regrade_gsm8k.py)),
which reads the final answer of a marker-less response instead of its last
number.

| Run | Correct | Accuracy |
|---|---:|---:|
| v0.1.1 (2026-09-02) | 1,278 / 1,319 | 96.9% |
| v0.1.0 (2026-08-31) | 1,282 / 1,319 | 97.2% |
| BF16-attention predecessor artifact | 1,280 / 1,319 | 97.0% |

Vision: 3840x2160 and 7680x4320 images at the model's 8,000-token budget
(7,994 prompt tokens) answer in 3.4 s and 3.0 s through the
OpenAI-compatible image input; the v0.1.0 bar-chart read-back check was
not repeated.

## Memory headroom (v0.1.3, 2026-09-02)

Measured with the engine's extend memory profiler (`SGLANG_EXTEND_MEM_PROFILE=1`,
caching-allocator counters on TP0) and a stress script that fills the pool
with long prompts and then sends a memory-hungry request (raw log in
`evidence/v0.1.3-headroom-study-20260902.txt`).

| Quantity | Value |
|---|---:|
| Weights per GPU (target + MTP draft) | 84.7 GB |
| KV pool, 499,712 tokens (FP8 DSA rows) | 4.40 GB |
| Mamba pool, 28 slots | 2.16 GB |
| Decode CUDA graphs | 1.5 GB |
| Live allocator memory at idle, pool 499,712 / 450,560 / 393,216 | 92.04 / 91.4 / 90.95 GiB |
| Allocator ceiling observed (64 MiB block failed) | 92.9 GiB |
| Text prefill chunk (4,096 tokens) peak transient | 727 MiB |
| Full-budget image encode (7,995 tokens) peak transient | 854 MiB |
| Three long requests resident | +0.2 GiB |

Stress shapes (all on v0.1.3-rc.1; "pass" = every request completes, no allocator warning, no restart):

| Shape | 499,712 (v0.1.2) | 393,216 | 450,560 |
|---|---|---|---|
| 3 long prompts resident + full-budget image | scheduler OOM in the vision encoder | pass | pass (x3) |
| 3 long prompts resident + cold 4k text | not run | pass | pass |
| single prompt at 97% of the pool + image | not run | pass | pass |
| 4 cold prompts at a quarter of the pool each | mamba assert (see below) | pass (x2) | pass |
| 3 long prompts + two images back to back | not run | pass | pass (x2) |
| 10 images with mixed text (42k tokens), alone and with 3 long prompts | not run | pass | pass |

The second crash at 499,712 was not memory: `AssertionError: Can not alloc
mamba cache` in `stash_chunked_request`. A cold long prefill ends with a
burst of about half a second in which every cached mamba state is
non-evictable (one slot per prefill chunk, all 28 in use), and a concurrent
chunk-boundary checkpoint found no slot. v0.1.3 skips that checkpoint and
gates admission on free plus evictable slots instead of asserting; the
skip fired twice across the whole study.

## Artifact reconstruction quality (producer receipts, unchanged)

| Tier | Metric | Value |
|---|---|---:|
| Routed experts (W4A16 NVFP4, K32 MSE) | aggregate relative L2 | 0.0851 |
| Routed experts | minimum matrix cosine | 0.9949 |
| FP8 [128,128]-block attention + shared experts (317 tensors) | aggregate relative L2 | 0.0225 |
| FP8 tier | minimum matrix cosine | 0.9996 |

## Known measurement gaps

- The 350k and 494k ladder rungs and the C4 fill ran on cached prefixes
  because of rung ordering; only the 200k rung and the engine-gate panel
  are cold prefill rates.
- Device headroom at full pool usage is effectively zero (see Capacity);
  a workload that allocates a large transient at that point would fail.
- One quantization configuration lesson is load-bearing enough to repeat:
  quantizing the MTP draft layer (layer 45) silently collapses speculative
  acceptance to zero while leaving outputs correct. The shipped producer
  keeps it BF16. See `CHANGELOG.md` for the full narrative.
