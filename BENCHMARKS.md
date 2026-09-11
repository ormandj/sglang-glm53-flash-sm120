# Measured results

## v0.4.3 measurements, 2026-09-11

### Standard fixed-MTP comparison

Measurements used a v0.4.3 validation build on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe. The GHCR image is built separately from the same pinned inputs and was not separately benchmarked. The configuration used W4A16 experts, FP8 KV, a 524,288-token shared device pool, four running requests, 4,096-token prefill chunks, 28 recurrent-state slots and 32 GB of HiCache per rank. Performance comparisons use fixed native MTP with three draft steps, top-k one and four verification tokens, with adaptive switching disabled. The launcher uses adaptive MTP for serving. The source inputs are pinned in [this release's stack lock](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.3/stack.lock.json).

| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 197.5 | 189.8 | 66.91 | 66.91 | 2.97 / 2.88 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 311.3 | 310.0 | 51.07 | 50.95 | 3.02 / 3.00 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 377.1 | 380.1 | 41.11 | 41.01 | 3.05 / 3.06 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 425.1 | 420.7 | 35.34 | 35.35 | 2.99 / 2.94 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,626.6 | 5,624.2 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 6,339.0 | 6,341.4 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 6,370.9 | 6,353.0 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 6,352.0 | 6,328.9 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 12.7-27.7 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.

Decode tok/s is aggregate output after MTP, including reasoning, across the stated number of concurrent requests. Forward passes/s counts target-model iterations. Every decode response reaches a deliberate 4,096-token output cap with `ignore_eos`; the post-answer tail can increase speculative acceptance, so this is not completed-answer throughput. Prefill tok/s is each cold request's prompt-token count divided by time to first token, summarized over five requests per length. These controlled measurements do not necessarily represent real-world performance.

Output tok/forward/request is a separate measured distribution, so multiplying table means need not reproduce the mean output rate. The following table also reports aggregate cold-prompt throughput over the complete cell.

| Cold prefill, C1 | Mean TTFT | Median TTFT | Aggregate prompt tok/s over the cell |
|---|---:|---:|---:|
| 8k | 1.458 s | 1.459 s | 5,619.4 |
| 32k | 5.171 s | 5.169 s | 6,336.2 |
| 64k | 10.289 s | 10.318 s | 6,368.8 |
| 128k | 20.597 s | 20.672 s | 6,350.6 |

### Comparison with v0.4.2

The v0.4.2 baseline uses the previously measured fixed-three-step panel with the same physical GPUs, power limit, driver, checkpoint, 524,288-token shared pool, serving settings, workloads, random seeds and analysis definitions. Both images use three draft steps, top-k one and four verification tokens, with adaptive switching disabled. Sequential repetitions within one startup per image do not isolate individual changes or establish statistical significance.

| Concurrency | Version | Mean output tok/s after MTP | Median output tok/s after MTP | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| C1 | v0.4.2 | 196.1 | 193.0 | 66.78 | 66.72 | 2.95 / 2.96 |
| C1 | v0.4.3 | 197.5 | 189.8 | 66.91 | 66.91 | 2.97 / 2.88 |
| C2 | v0.4.2 | 306.0 | 306.8 | 50.99 | 50.89 | 3.00 / 3.02 |
| C2 | v0.4.3 | 311.3 | 310.0 | 51.07 | 50.95 | 3.02 / 3.00 |
| C3 | v0.4.2 | 369.0 | 370.2 | 41.08 | 41.03 | 2.99 / 3.05 |
| C3 | v0.4.3 | 377.1 | 380.1 | 41.11 | 41.01 | 3.05 / 3.06 |
| C4 | v0.4.2 | 411.0 | 408.4 | 35.38 | 35.16 | 2.90 / 2.91 |
| C4 | v0.4.3 | 425.1 | 420.7 | 35.34 | 35.35 | 2.99 / 2.94 |

Fixed native MTP: 3 draft steps, top-k 1, 4 verification tokens; adaptive switching disabled. Five repetitions per version and concurrency, 17,408-20,480 average context tokens, 12.7-29.3 seconds per measured window. Output is aggregate across the cohort. All requests use a fixed 4,096-token ignore_eos window including a post-answer tail; these are controlled measurements, not completed-answer throughput.

| Cold prefill, C1 | v0.4.2 mean / median prompt tok/s | v0.4.3 mean / median prompt tok/s | Mean / median change |
|---|---:|---:|---:|
| 8k, 5 requests per version | 5,647.7 / 5,659.0 | 5,626.6 / 5,624.2 | -0.37% / -0.61% |
| 32k, 5 requests per version | 6,352.0 / 6,359.6 | 6,339.0 / 6,341.4 | -0.20% / -0.29% |
| 64k, 5 requests per version | 6,361.1 / 6,345.1 | 6,370.9 / 6,353.0 | +0.15% / +0.13% |
| 128k, 5 requests per version | 6,346.7 / 6,319.5 | 6,352.0 / 6,328.9 | +0.08% / +0.15% |

Compared with v0.4.2, mean output rates changed by +0.67% at C1, +1.74% at C2, +2.19% at C3 and +3.44% at C4; median output changes ranged from -1.67% to +3.00%. Mean and median target forward rates differed by at most 0.54%, and cold-prefill rates by at most 0.62%. Output-rate changes also reflect speculative acceptance in the fixed window, including its post-answer tail. Five sequential repetitions within one startup per image do not establish an isolated speedup or statistical significance.

### Correctness coverage

The measured validation build passed CPU and GPU checks for sparse-attention layout selection, masked reads, GLM head-count specializations, graph replay, cache transfers, MHC, W4A16 MoE and scheduler ownership. HiCache coverage included packed transfers and mixed target/draft sidecars under Compute Sanitizer, with zero reported sanitizer errors. Serving checks covered five forced host restores followed by device replays and approximately 400k-token positional recall through cold, device and host-cache paths. These checks exercise the changed paths and carried integrations on the stated two-GPU platform.

### Answer quality and startup

The launcher uses adaptive three/five-step MTP for serving. The following quality and reliability checks used that serving configuration; published performance comparisons use the fixed-MTP standard above.

The adaptive-serving GSM8K run completed all 1,319 requests without request errors or output caps. With the unchanged 16,384-token budget and default reasoning, the GLM-aware extractor matched 1,283 answers and the stock extractor matched 1,176. The matched v0.4.2 run had 1,279 GLM-aware matches, including 1,277 from completed responses, and 1,166 stock matches; two responses exhausted the budget while reasoning. These single runs do not establish quality improvement or equivalence.

The unchanged long-context GSM8K workload scored 143/150 at approximately 73k tokens and 142/150 at approximately 400k tokens, compared with 144/150 and 143/150 for v0.4.2. These checks used a 2,048-token budget and `reasoning_effort: low`; the harness retained extracted answers but not finish reasons, so completion status is unknown.

On the first boot of the measured validation build, the first short thinking response finished in 0.826 seconds and the first sampled response in 0.620 seconds. A 7680×4320 image followed by four simultaneous independent cold 4k-token requests completed without an out-of-memory failure or restart, both after startup and after the quality workload. A 523,787-token prompt also ran with a deliberate 256-token output window. The 524,288-token device pool is shared across requests; it is not a per-request reservation.

The reusable workloads and measurement tools are in [bench/](bench/). Other GPU pairs and tensor-parallel sizes have not been measured.
