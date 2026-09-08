# Measured results

## v0.4.0 measurements, 2026-09-08

### Standard fixed-MTP comparison

Measurements used a v0.4.0 validation build on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe. The GHCR image was built separately from the same pinned inputs and was not separately benchmarked. The configuration used W4A16 experts, FP8 KV, a 524,288-token shared device pool, four running requests, 4,096-token prefill chunks, 28 recurrent-state slots and 32 GB of HiCache per rank. These comparison runs use fixed native MTP with three draft steps, top-k one and four verification tokens, with adaptive switching disabled. The measured source inputs are pinned in [this release's stack lock](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.0/stack.lock.json).

| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 191.0 | 197.1 | 67.13 | 67.21 | 2.87 / 2.96 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 300.4 | 296.8 | 51.21 | 51.05 | 2.93 / 2.90 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 365.5 | 369.9 | 41.46 | 41.26 | 2.91 / 2.97 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 415.3 | 418.1 | 35.41 | 35.31 | 2.92 / 2.91 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,642.3 | 5,673.0 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 6,352.2 | 6,346.8 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 6,375.0 | 6,356.6 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 6,349.4 | 6,326.7 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 14.3-29.0 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.

Decode rows report aggregate output after MTP across the active cohort, including reasoning and content. Forward passes/s counts target-model iterations. Each cell has five repetitions of a 4,096-token `ignore_eos` response after approximately 16k prompt tokens; rates cover the selected steady decode window. The post-answer tail can have higher speculative acceptance than the natural answer. Output tok/forward/request is a separate measured distribution, so multiplying table means need not reproduce the mean output rate.

Prefill rows report the mean and median of each request's prompt tokens divided by time to first token, over five cold C1 requests per length. The following table also retains aggregate prompt throughput over the complete cell. These controlled measurements do not necessarily represent real-world performance.

| Cold prefill, C1 | Mean TTFT | Median TTFT | Aggregate prompt tok/s over the cell |
|---|---:|---:|---:|
| 8k | 1.454 s | 1.446 s | 5,635.6 |
| 32k | 5.161 s | 5.165 s | 6,348.8 |
| 64k | 10.283 s | 10.312 s | 6,373.1 |
| 128k | 20.606 s | 20.679 s | 6,347.9 |

### Comparison with v0.3.2

Both versions were measured again with fixed three-step MTP using identical workloads, random seeds and analysis definitions. The images retain their respective device-pool capacities: v0.4.0 increases the pool from 450,560 to 524,288 tokens and changes the prefill implementation. Sequential repetitions within one startup per image do not isolate an individual code change or establish statistical significance.

| Concurrency | Version | Mean output tok/s after MTP | Median output tok/s after MTP | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| C1 | v0.3.2 | 203.7 | 191.5 | 66.93 | 66.90 | 3.06 / 2.90 |
| C1 | v0.4.0 | 191.0 | 197.1 | 67.13 | 67.21 | 2.87 / 2.96 |
| C2 | v0.3.2 | 317.5 | 304.4 | 51.16 | 51.19 | 3.07 / 3.00 |
| C2 | v0.4.0 | 300.4 | 296.8 | 51.21 | 51.05 | 2.93 / 2.90 |
| C3 | v0.3.2 | 359.4 | 359.4 | 41.34 | 41.39 | 2.91 / 2.91 |
| C3 | v0.4.0 | 365.5 | 369.9 | 41.46 | 41.26 | 2.91 / 2.97 |
| C4 | v0.3.2 | 411.9 | 407.2 | 35.63 | 35.70 | 2.89 / 2.86 |
| C4 | v0.4.0 | 415.3 | 418.1 | 35.41 | 35.31 | 2.92 / 2.91 |

Fixed native MTP: 3 draft steps, top-k 1, 4 verification tokens; adaptive switching disabled. Five repetitions per version and concurrency, 17,408-20,480 average context tokens, 11.7-29.0 seconds per measured window. Output is aggregate across the cohort. All requests use a fixed 4,096-token ignore_eos window including a post-answer tail; these are controlled measurements, not completed-answer throughput.

| Cold prefill, C1 | v0.3.2 mean / median prompt tok/s | v0.4.0 mean / median prompt tok/s | Mean / median change |
|---|---:|---:|---:|
| Cold prefill 8k, C1, 5 requests | 5,300.8 / 5,324.1 | 5,642.3 / 5,673.0 | +6.44% / +6.55% |
| Cold prefill 32k, C1, 5 requests | 5,929.3 / 5,938.3 | 6,352.2 / 6,346.8 | +7.13% / +6.88% |
| Cold prefill 64k, C1, 5 requests | 5,982.1 / 5,967.4 | 6,375.0 / 6,356.6 | +6.57% / +6.52% |
| Cold prefill 128k, C1, 5 requests | 5,960.9 / 5,941.4 | 6,349.4 / 6,326.7 | +6.52% / +6.49% |

### Answer quality and startup

The launcher uses adaptive three/five-step MTP for serving. The following quality and reliability checks used that serving configuration; published performance comparisons use the fixed-MTP standard above.

Across all 1,319 GSM8K questions, 1286 answers (97.5%) were scored correct by the GLM-aware answer extractor. The stock AIPerf extractor scored 1179 (89.4%) on the same responses. All requests succeeded; one response reached the 16,384-token output cap and remains in the denominator; the other 1,318 ended naturally. Results depend on the stated answer extraction method. One run per image does not establish a quality improvement or equivalence.

The 150-question long-context controls scored 143/150 at about 73k tokens and 145/150 at about 400k.

The first sampled chat completed in 0.640 seconds with a 0.293-second time to first token. A 7680x4320 image followed by four independent cold text requests completed without a server restart. A 523,787-token prompt also completed its 256-token output window. The 524,288-token pool is shared across concurrent requests; it does not provide four independent 524,288-token contexts.

The reusable workloads and measurement tools are in [bench/](bench/). Other GPU pairs and tensor-parallel sizes have not been measured.
