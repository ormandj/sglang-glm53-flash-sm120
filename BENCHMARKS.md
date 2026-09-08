# Measured results

## v0.4.1 measurements, 2026-09-08

### Standard fixed-MTP comparison

Measurements used a v0.4.1 validation build on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe. The GHCR image was built separately from the same pinned inputs and was not separately benchmarked. The configuration used W4A16 experts, FP8 KV, a 524,288-token shared device pool, four running requests, 4,096-token prefill chunks, 28 recurrent-state slots and 32 GB of HiCache per rank. Performance comparisons use fixed native MTP with three draft steps, top-k one and four verification tokens, with adaptive switching disabled. The launcher uses adaptive MTP for serving. The source inputs are pinned in [this release's stack lock](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.1/stack.lock.json).

| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 181.5 | 174.7 | 67.11 | 67.04 | 2.73 / 2.61 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 312.1 | 293.5 | 51.12 | 51.14 | 3.05 / 2.84 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 371.6 | 379.0 | 41.18 | 40.98 | 2.99 / 3.01 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 410.4 | 406.5 | 35.66 | 35.60 | 2.88 / 2.84 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,616.6 | 5,629.8 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 6,344.2 | 6,338.5 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 6,377.1 | 6,371.1 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 6,359.6 | 6,336.7 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 14.3-29.7 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.

Decode rows report aggregate output after MTP across the active cohort, including reasoning and content. Forward passes/s counts target-model iterations. Each cell has five repetitions of a 4,096-token `ignore_eos` response after approximately 16k prompt tokens; all responses reach that deliberate output cap, and rates cover the selected steady decode window. The post-answer tail can have higher speculative acceptance than the natural answer. Output tok/forward/request is a separate measured distribution, so multiplying table means need not reproduce the mean output rate.

Prefill rows report the mean and median of each request's prompt tokens divided by time to first token, over five cold C1 requests per length. The following table also retains aggregate prompt throughput over the complete cell. These controlled measurements do not necessarily represent real-world performance.

| Cold prefill, C1 | Mean TTFT | Median TTFT | Aggregate prompt tok/s over the cell |
|---|---:|---:|---:|
| 8k | 1.461 s | 1.457 s | 5,609.3 |
| 32k | 5.167 s | 5.172 s | 6,340.8 |
| 64k | 10.279 s | 10.288 s | 6,375.4 |
| 128k | 20.573 s | 20.646 s | 6,357.9 |

### Comparison with v0.4.0

The v0.4.0 baseline uses the previously measured fixed-three-step panel with the same hardware, checkpoint, 524,288-token shared pool, serving settings, workloads, random seeds and analysis definitions. Both images use three draft steps, top-k one and four verification tokens, with adaptive switching disabled. Sequential repetitions within one startup per image do not isolate individual changes or establish statistical significance.

| Concurrency | Version | Mean output tok/s after MTP | Median output tok/s after MTP | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| C1 | v0.4.0 | 191.0 | 197.1 | 67.13 | 67.21 | 2.87 / 2.96 |
| C1 | v0.4.1 | 181.5 | 174.7 | 67.11 | 67.04 | 2.73 / 2.61 |
| C2 | v0.4.0 | 300.4 | 296.8 | 51.21 | 51.05 | 2.93 / 2.90 |
| C2 | v0.4.1 | 312.1 | 293.5 | 51.12 | 51.14 | 3.05 / 2.84 |
| C3 | v0.4.0 | 365.5 | 369.9 | 41.46 | 41.26 | 2.91 / 2.97 |
| C3 | v0.4.1 | 371.6 | 379.0 | 41.18 | 40.98 | 2.99 / 3.01 |
| C4 | v0.4.0 | 415.3 | 418.1 | 35.41 | 35.31 | 2.92 / 2.91 |
| C4 | v0.4.1 | 410.4 | 406.5 | 35.66 | 35.60 | 2.88 / 2.84 |

Fixed native MTP: 3 draft steps, top-k 1, 4 verification tokens; adaptive switching disabled. Five repetitions per version and concurrency, 17,408-20,480 average context tokens, 14.3-29.7 seconds per measured window. Output is aggregate across the cohort. All requests use a fixed 4,096-token ignore_eos window including a post-answer tail; these are controlled measurements, not completed-answer throughput.

| Cold prefill, C1 | v0.4.0 mean / median prompt tok/s | v0.4.1 mean / median prompt tok/s | Mean / median change |
|---|---:|---:|---:|
| Cold prefill 8k, C1, 5 requests | 5,642.3 / 5,673.0 | 5,616.6 / 5,629.8 | -0.46% / -0.76% |
| Cold prefill 32k, C1, 5 requests | 6,352.2 / 6,346.8 | 6,344.2 / 6,338.5 | -0.13% / -0.13% |
| Cold prefill 64k, C1, 5 requests | 6,375.0 / 6,356.6 | 6,377.1 / 6,371.1 | +0.03% / +0.23% |
| Cold prefill 128k, C1, 5 requests | 6,349.4 / 6,326.7 | 6,359.6 / 6,336.7 | +0.16% / +0.16% |

Observed mean and median forward rates and cold-prefill rates differ by less than 1% between these runs. Output rates vary in both directions; these results do not demonstrate an overall throughput improvement.

### Correctness coverage

The measured validation build passed CPU and GPU checks for cache transfer and compressed index ownership, sparse attention and masked reads, CUDA-graph buffer lifetimes, KDA, MHC, MoE routing, vision offload and request scheduling. HiCache coverage included mixed FP8-target/BF16-draft storage and byte/state round trips under Compute Sanitizer, with zero reported sanitizer errors. These checks exercise the changed paths and carried integrations on the stated two-GPU platform.

### Answer quality and startup

The launcher uses adaptive three/five-step MTP for serving. The following quality and reliability checks used that serving configuration; published performance comparisons use the fixed-MTP standard above.

Across all 1,319 GSM8K questions, 1,282 answers (97.2%) were scored correct by the GLM-aware answer extractor. The stock AIPerf extractor scored 1,172 (88.9%) on the same responses. All requests succeeded in normal-EOS mode; none reached the 16,384-token output cap. Results depend on the stated answer extraction method. One run per image does not establish a quality improvement or equivalence.

The 150-question long-context controls scored 144/150 at about 73k tokens and 145/150 at about 400k.

The first sampled chat completed in 0.550 seconds with a 0.257-second time to first token. A 7680x4320 image followed by four independent cold text requests completed without a server restart. A 523,787-token prompt also completed its deliberately capped 256-token output window. The 524,288-token pool is shared across concurrent requests; it does not provide four independent 524,288-token contexts.

The reusable workloads and measurement tools are in [bench/](bench/). Other GPU pairs and tensor-parallel sizes have not been measured.
