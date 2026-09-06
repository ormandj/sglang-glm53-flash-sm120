# Measured results

## v0.3.2 qualification, 2026-09-06

The internal candidate `v0.3.2-rc.1` was qualified and promoted without rebuilding to `v0.3.2` at `sha256:6b5ed4f7f8e6a56076f8446a11240dd1a4d9d49fdf62c07ad345026678890dc5`. Hardware: two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, TP2/EP1 over PCIe. The profile retains W4A16 experts, FP8 KV, a 450,560-token pool, four running requests, 4,096-token prefill chunks and 32 GB of HiCache per rank. These measurements apply to that exact internal digest; the independently built public image has separate provenance.

SGLang is based on official main `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110`, after GLM support merged as `97c6978369ac1e04c91fcc01c98acc25129a6000`. FlashInfer is based on main `6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f`. Both were freshly checked for this rebuild. The combined carried integration was reconciled with main, superseded metadata experiments were removed, and active #38213 fusion and the reviewed route-prefix implementation were retained. Exact source trees and patch hashes are in `stack.lock.json`.

| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 215.5 | 202.6 | 60.77 | 61.38 | 3.60 / 3.42 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 290.2 | 293.8 | 49.11 | 49.16 | 3.02 / 3.08 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 360.3 | 356.6 | 39.20 | 38.98 | 3.11 / 3.12 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 406.9 | 410.2 | 33.31 | 33.62 | 3.06 / 3.03 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,193.5 | 5,211.0 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 5,877.4 | 5,860.4 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 5,921.4 | 5,907.2 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 5,923.1 | 5,903.7 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 10.7-29.7 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.

Decode rows report aggregate output after MTP across the active C1-C4 cohort, including reasoning and content. Forward passes/s counts target-model iterations. Each cell has five repetitions of a 4,096-token `ignore_eos` response after approximately 16k prompt tokens; rates cover the analyzer-selected steady decode window. The post-answer tail can have higher speculative acceptance than the natural answer. Output tok/forward/request is retained separately from the rate distributions, so multiplying table means need not reproduce the mean output rate.

Prefill rows report the mean and median of each request's prompt tokens divided by time to first token, over five cold C1 requests per length. The latency table also preserves aggregate prompt throughput over the complete cell. Prefill is input processing; MTP applies to decode. These controlled measurements do not necessarily represent real-world performance.

| Cold prefill, C1 | Mean TTFT | Median TTFT | Aggregate prompt tok/s over the cell |
|---|---:|---:|---:|
| 8k | 1.580 s | 1.574 s | 5,186.6 |
| 32k | 5.578 s | 5.594 s | 5,875.3 |
| 64k | 11.070 s | 11.096 s | 5,920.0 |
| 128k | 22.089 s | 22.160 s | 5,921.8 |

All 24 original engine cell analyzers passed and all 24 resolved workloads match the retained v0.3.1 baseline, excluding only artifact output directories. The raw adaptive acceptance-rate gauge exceeded one in C1/r03; it is retained as diagnostic telemetry and is not an acceptance probability. No cell rerun or summary repair was needed.

### Fixed-acceptance and capped varied-prompt probes

The repetitive-ledger C1 probe uses six 2,048-output-token repetitions per context after a 150-second soak. Both throughput columns cover its decode plateau; accepted output per forward is the ratio of the two fitted rates.

| Context tokens | Mean output tok/s after MTP | Median output tok/s after MTP | Mean forwards/s | Median forwards/s | Output tok/forward, mean / median |
|---|---:|---:|---:|---:|---:|
| 1024 | 155.5 | 154.2 | 67.53 | 67.59 | 2.30 / 2.29 |
| 19000 | 160.2 | 160.1 | 67.39 | 67.46 | 2.38 / 2.37 |

The varied-prompt C1 probe enables normal EOS and uses essay, code-generation and Q&A prompts, two repetitions each, greedy sampling with thinking enabled and a 4,096-token cap. All six responses reached that cap, so these are budget-limited decode measurements, not completed-answer rates. It measures from first to last streamed token, including reasoning. Forward counters are sampled every 250 ms, which limits boundary precision for short replies. Finish reasons: essay: length, length; codegen: length, length; qa: length, length. A `length` result is a capped response, not a completed answer; all results remain in the table.

| Workload | Mean output tok/s after MTP | Median output tok/s after MTP | Mean forwards/s | Median forwards/s | Output tok/forward, mean / median |
|---|---:|---:|---:|---:|---:|
| essay | 157.9 | 157.9 | 66.65 | 66.65 | 2.37 / 2.37 |
| codegen | 164.7 | 164.7 | 65.25 | 65.25 | 2.52 / 2.52 |
| qa | 166.9 | 166.9 | 64.75 | 64.75 | 2.58 / 2.58 |

These probes and the engine panel describe this run; they do not establish statistical significance or isolate any individual patch's effect. [Raw engine comparison](evidence/v0.3.2/engine-comparison.json) and all probe repetitions are retained.

### Quality and stability

Full GSM8K scored 1,279/1,319 (97.0%) with the unchanged GLM-aware grader and 1,171/1,319 (88.8%) with pinned AIPerf, with zero request errors. One response reached the 16,384-token budget and remains in both denominators. The graders disagreed on 109 pinned-fail/GLM-pass responses and one in the reverse direction. The reverse disagreement contains the correct answer but the GLM-aware extractor selects a later bold time label; neither grader was changed.

The 73k and 400k controls scored 142/150 and 144/150. Prefill rescoring of a 768-token continuation at about 400k context agreed on all 767 next-token choices. Cold and device-cached recall preserved seven ordered markers at 408k. All five confirmed host-load cycles preserved the expected tool decision, and the separate forced-host 408k ordered-marker oracle passed all four stages. The 408k forced-host strict-text check differed in optional `MEMORY-CHECK:` labels after restoration; its original failure is retained alongside the passing ordered-marker oracle. No byte-identical generation claim is made.

Against retained v0.3.1 responses, gsm8k lost 10 and gained 9; longctx-73k lost 3 and gained 2; longctx-400k lost 2 and gained 1. These are single-run paired observations and establish neither quality improvement nor equivalence. No baseline campaign was repeated.

The exact-image CPU/GPU matrix and installed FlashInfer tests passed. FlashInfer's route-packing validation passed 124 tests and 1,720 cases under each of memcheck, racecheck and synccheck, with zero errors or hazards. First-boot sampled chat completed in 0.690 seconds (TTFT 0.470 seconds); both startup and final 7680x4320-image/independent-cold-C4 checks passed. The final qualification pod was Ready with zero restarts and only its two serving GPU processes. [Receipts and limitations](evidence/v0.3.2/README.md).

## v0.3.1 qualification, 2026-09-06

The internal candidate `v0.3.1-rc.1` is qualified and promoted without rebuilding as `v0.3.1` at `sha256:5c2c6fb8f5616d3b451d45d944f56248f0e81c6cc403b80aa8c296f8391b5e2d`. These measurements apply to that exact internal image on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, TP2/EP1 over PCIe. The profile retains W4A16 experts, FP8 KV, a 450,560-token pool, four running requests, 4,096-token prefill chunks and 32 GB of HiCache per rank.

SGLang remains main `febb360519` plus the recorded integration, including GLM support carried from the pre-merge #36507 branch. FlashInfer remains main `6c14bbd5ff` plus the carried changes and the small route-prefix optimization. The owner deferred a post-merge SGLang refresh to a separate rebuild.

The exact-image GPU/CPU matrix, sampled startup, 7680x4320 image followed by four independent cold requests, full GSM8K, long-context and forced-host checks completed. Full GSM8K scored 1,181/1,319 (89.5%) with pinned AIPerf and 1,280/1,319 (97.0%) with the unchanged GLM-aware grader, with zero request errors. Two responses exhausted the 16,384-token budget and remain in both denominators. Grader disagreements are 100 pinned-fail/GLM-pass and one in the reverse direction.

The 73k and 400k controls scored 143/150 and 145/150. Against the existing v0.3.0 baseline, full GSM8K lost six previously correct answers and gained seven; 73k lost one and gained five; 400k lost two and gained six. These single-run paired changes establish neither quality improvement nor equivalence. Prefill rescoring of the 768-token continuation at 400k context agreed on all 767 next-token choices. Cold/device recall and the separate forced-host 408k probe preserved all seven ordered markers. All five reconstructed-request host cycles preserved the expected tool decision. The 408k strict-text check failed only because optional `MEMORY-CHECK:` labels changed after restoration; its original failure and the pre-existing marker-oracle pass are both retained.

| Engine cell | Mean forwards/s | Median forwards/s | Sample CV |
|---|---:|---:|---:|
| C1, n=5 | 62.61 | 64.28 | 5.66% |
| C2, n=5 | 48.91 | 48.28 | 3.99% |
| C3, n=5 | 38.75 | 40.01 | 9.03% |
| C4, n=5 | 34.24 | 35.33 | 5.70% |

| Cold prefill | Prompt tok/s | Median TTFT |
|---|---:|---:|
| 8k, five requests | 5,291 | 1.545 s |
| 32k, five requests | 5,944 | 5.501 s |
| 64k, five requests | 5,970 | 11.012 s |
| 128k, five requests | 5,935 | 22.130 s |

All 24 cell analyzers passed, with configs byte-identical to the retained baseline. Final summarization initially failed because an adaptive speculative-acceptance gauge reached 1.3583, exceeding an incorrect fixed 1.25 bound. Source and raw samples show interval-wide accepted drafts divided by the currently active draft width. The corrected summarizer retains and flags this diagnostic ratio; it changes no cell verdict, workload, counter or measured rate. Codex and Claude approved the exact correction and its 26 focused tests passed. The original failure and data are preserved separately from the reviewed summary, with full hash provenance. All adaptive acceptance-rate gauge statistics, including values below one, can be biased and are not reliable probabilities. Fixed-window engine rates are engineering comparisons, not expected application throughput.

The candidate-only fixed-acceptance follow-up measured 67.406 ± 0.338 forwards/s at 1k and 66.988 ± 0.511 at 19k (six repetitions, mean ± sample SD). The initial candidate means were 67.073 and 66.187, versus baseline 67.360 and 67.042. The earlier 19k decrease did not recur at the same size; no zero-effect or causal clock claim is made. The original matched natural-EOS C2/C4 results and full private fixture receipts remain in the primary qualification repository. No baseline campaign was repeated.

The retained paired engine comparison measured v0.3.0 / v0.3.1 mean forwards/s of 61.67 / 62.61 at C1, 43.95 / 48.91 at C2, 36.66 / 38.75 at C3 and 32.04 / 34.24 at C4. These sequential, adaptive-path measurements are descriptive and do not establish statistical significance or isolate the route-prefix change. [All repetitions and comparison](evidence/v0.3.1/engine-comparison.json).

The final qualification pod was Ready with zero restarts and only its two scheduler GPU processes. Exact-candidate receipts, both graders, paired IDs and the reviewed engine analysis are in [the release evidence](evidence/v0.3.1/README.md).

## Historical measurements

Three measurement sets are recorded here: the v0.2.0-rc.1 set below, then the earlier two. The v0.1.1 set was measured on
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

## v0.2.0-rc.1 (2026-09-03): performance candidate on the v0.1.4 bases

Internal digest `sha256:020d252ab9754346cce2f3ae7f869e857ead1b0663387d16c958e03cfd1e3bdb`,
pool and context 450,560, mamba 28, HiCache 32 GB, cache schema v56. Changes: PCIe IPC
all-reduce wired in, KDA gate-projection fusion, upstream's experimental device-side DSA kpool
metadata, in-place FP8
blockwise lm_head shared by target and draft. Method notes and every kept or dropped experiment
are in the primary repository (`evidence/v0.1.4-c1-perf-experiments-20260902.md`, receipt
`evidence/v0.2.0-rc.1-validation-20260903.txt`).

### Decode (engine gates, five analyzer-valid repetitions per cell)

| Cell | OLS tok/s per repetition | Mean tok/s | Median tok/s | Forwards/s per repetition | Mean fwd/s | Accepted tok/fwd/req (mean) |
|---|---|---:|---:|---|---:|---:|
| C1 | 283.1 / 290.4 / 296.6 / 174.2 / 276.2 | 264.1 | 283.1 | 52.4 / 51.6 / 50.6 / 66.0 / 52.3 | 54.6 | 4.9 |
| C2 | 306.9 / 271.6 / 348.1 / 278.9 / 329.0 | 306.9 | 306.9 | 38.1 / 41.3 / 37.3 / 42.1 / 36.9 | 39.1 | 3.9 |
| C3 | 387.1 / 349.5 / 522.1 / 372.3 / 447.6 | 415.7 | 387.1 | 29.8 / 33.6 / 31.0 / 30.9 / 30.1 | 31.1 | 4.5 |
| C4 | 366.5 / 409.9 / 380.5 / 389.3 / 385.8 | 386.4 | 385.8 | 32.0 / 31.5 / 31.6 / 30.4 / 31.2 | 31.3 | 3.1 |
| C4 (v0.1.4, same gate) | 352.6 / 417.6 / 360.6 / 341.2 / 387.7 | 371.9 | 360.6 | 28.5 / 27.7 / 29.5 / 31.0 / 27.2 | 28.8 | 3.3 (3.1 median) |
| C1 (v0.1.4, same gate, 2026-09-03) | 150.1 / 153.2 / 162.7 / 145.8 / 171.1 | 156.6 | 153.2 | 55.4 / 55.4 / 53.1 / 56.4 / 52.2 | 54.5 | 2.9 |
| C2 (v0.1.4, same gate) | 256.8 / 236.5 / 223.7 / 233.6 / 235.0 | 237.1 | 235.0 | 36.9 / 38.6 / 41.0 / 39.9 / 39.7 | 39.2 | 3.0 |
| C3 (v0.1.4, same gate) | 341.9 / 322.4 / 304.0 / 300.2 / 326.7 | 319.0 | 322.4 | 27.9 / 31.4 / 32.0 / 31.5 / 30.3 | 30.6 | 3.5 |

Concurrency-1 probe (`benchmarks/profile/c1_bench.py` in the primary repository: ledger prompt,
2,048 output tokens, 150 s thermal soak, six runs, SM clock/power/temperature logged; both images on the same protocol):

| Context | v0.1.4 fwd/s / tok/s | v0.2.0-rc.1 fwd/s / tok/s | Accepted tok/fwd (v0.1.4 / v0.2.0) |
|---|---:|---:|---:|
| 1k | 56.2 +- 0.9 / 130 | 67.1 +- 0.4 / 154 | 2.35 / 2.31 |
| 19k | 56.1 +- 0.7 / 132 | 66.5 +- 0.3 / 155 | 2.37 / 2.34 |

The engine-gate rates above are synthetic fixed-window signals with path-dependent acceptance;
repetitions are prompt-path subsamples, not independent replicates or an application throughput.
At C1-C3 the accepted column (a mean) depends on where each greedy answer ends inside the fixed
4,096-token `ignore_eos` window: past the end token the drafter accepts 5.8-6.0 of 6. The
like-for-like engine comparison is the step rate at equal acceptance, measured at concurrency 1
only (the probe above, +19% with about 3% cross-boot noise); no equal-acceptance estimate was
measured for two or three requests.

### Prefill (five cold requests per length, C1)

| Shape | Rate | Median TTFT |
|---|---:|---:|
| 8k | 5,245 tok/s | 1.56 s |
| 32k | 5,871 tok/s | 5.60 s |
| 64k | 5,890 tok/s | 11.13 s |
| 128k | 5,893 tok/s | 22.22 s |

v0.1.4-rc.2 on the same panel the same night: 8k 5,228 / 32k 5,867 / 64k 5,609 / 128k 5,891 tok/s.

### Quality and stability

| Check | Result |
|---|---|
| GSM8K 300-question subset, greedy, C4, shipped profile | 296/300 regraded (v0.1.4-rc.2 on the same run: 296/300) |
| FP8 lm_head vs BF16 head, first-token top-10 logprobs, 300 cold prompts | at the BF16-vs-BF16 noise floor (top-1 agreement 89.0% vs 89.7%, KL 0.020 vs 0.022 nats; measured on the pre-build overlay of the same code; on the immutable images, rc.1 FP8 head vs the v0.1.4 image's BF16 head: 79.7% and 0.054, against a control of the v0.1.4 image's BF16 head vs the pre-build overlay's BF16 head (different code) at 82.7% and 0.054: KL equal, top-1 3.0 points lower, consistent with the control) |
| 4 x 18k distinct concurrent (x2) | 4 decode together |
| 3 x 146k concurrent + full-budget 3840x2160 image | pass |
| 4 x 111.6k cold concurrent (cache flushed first) | pass, TTFT 19.3 s |
| 10 x 4K images alone; 10 x 4K images with 3 x 128k resident | pass, 2.3-3.5 s each |
| image then four cold 4k prompts | pass |
| First sampled thinking chat after ready | 1.4 s |
| Restarts, OOM, allocator warnings | 0 / 0 / 0 on these workloads. Known hazard (also on v0.1.4): a `/generate` request with input logprobs over a long prompt OOMs the scheduler's fp32 logits and restarts the container; keep such requests short or off |

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
| C4 (v0.1.3, pool 450,560) | 339.5 / 342.2 / 355.4 / 372.1 / 379.1 | 357.7 | 355.4 | 30.7 / 30.8 / 30.0 / 25.7 / 27.5 | 28.9 | 2.9 |
| C4 (v0.1.4, pool 450,560, refreshed bases) | 352.6 / 417.6 / 360.6 / 341.2 / 387.7 | 371.9 | 360.6 | 28.5 / 27.7 / 29.5 / 31.0 / 27.2 | 28.8 | 3.1 |

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
