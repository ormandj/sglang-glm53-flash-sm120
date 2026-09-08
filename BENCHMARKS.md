# Measured results

## v0.4.2 measurements, 2026-09-08

### Standard fixed-MTP comparison

Measurements used a v0.4.2 validation build on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe. The GHCR image is built separately from the same pinned inputs and was not separately benchmarked. The configuration used W4A16 experts, FP8 KV, a 524,288-token shared device pool, four running requests, 4,096-token prefill chunks, 28 recurrent-state slots and 32 GB of HiCache per rank. Performance comparisons use fixed native MTP with three draft steps, top-k one and four verification tokens, with adaptive switching disabled. The launcher uses adaptive MTP for serving. The source inputs are pinned in [this release's stack lock](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.2/stack.lock.json).

| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 196.1 | 193.0 | 66.78 | 66.72 | 2.95 / 2.96 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 306.0 | 306.8 | 50.99 | 50.89 | 3.00 / 3.02 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 369.0 | 370.2 | 41.08 | 41.03 | 2.99 / 3.05 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 411.0 | 408.4 | 35.38 | 35.16 | 2.90 / 2.91 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,647.7 | 5,659.0 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 6,352.0 | 6,359.6 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 6,361.1 | 6,345.1 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 6,346.7 | 6,319.5 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 13.3-29.3 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.

Decode tok/s is aggregate output after MTP, including reasoning, across the stated number of concurrent requests. Forward passes/s counts target-model iterations. Every decode response reaches a deliberate 4,096-token output cap with `ignore_eos`; the post-answer tail can increase speculative acceptance, so this is not completed-answer throughput. Prefill tok/s is each cold request's prompt-token count divided by time to first token, summarized over five requests per length. These controlled measurements do not necessarily represent real-world performance.

Output tok/forward/request is a separate measured distribution, so multiplying table means need not reproduce the mean output rate. The following table also reports aggregate cold-prompt throughput over the complete cell.

| Cold prefill, C1 | Mean TTFT | Median TTFT | Aggregate prompt tok/s over the cell |
|---|---:|---:|---:|
| 8k | 1.453 s | 1.450 s | 5,640.3 |
| 32k | 5.161 s | 5.155 s | 6,349.3 |
| 64k | 10.305 s | 10.331 s | 6,358.6 |
| 128k | 20.615 s | 20.702 s | 6,345.2 |

### Comparison with v0.4.1

The v0.4.1 baseline uses the previously measured fixed-three-step panel with the same physical GPUs, power limit, driver, checkpoint, 524,288-token shared pool, serving settings, workloads, random seeds and analysis definitions. Both images use three draft steps, top-k one and four verification tokens, with adaptive switching disabled. Sequential repetitions within one startup per image do not isolate individual changes or establish statistical significance.

| Concurrency | Version | Mean output tok/s after MTP | Median output tok/s after MTP | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| C1 | v0.4.1 | 181.5 | 174.7 | 67.11 | 67.04 | 2.73 / 2.61 |
| C1 | v0.4.2 | 196.1 | 193.0 | 66.78 | 66.72 | 2.95 / 2.96 |
| C2 | v0.4.1 | 312.1 | 293.5 | 51.12 | 51.14 | 3.05 / 2.84 |
| C2 | v0.4.2 | 306.0 | 306.8 | 50.99 | 50.89 | 3.00 / 3.02 |
| C3 | v0.4.1 | 371.6 | 379.0 | 41.18 | 40.98 | 2.99 / 3.01 |
| C3 | v0.4.2 | 369.0 | 370.2 | 41.08 | 41.03 | 2.99 / 3.05 |
| C4 | v0.4.1 | 410.4 | 406.5 | 35.66 | 35.60 | 2.88 / 2.84 |
| C4 | v0.4.2 | 411.0 | 408.4 | 35.38 | 35.16 | 2.90 / 2.91 |

Fixed native MTP: 3 draft steps, top-k 1, 4 verification tokens; adaptive switching disabled. Five repetitions per version and concurrency, 17,408-20,480 average context tokens, 13.3-29.7 seconds per measured window. Output is aggregate across the cohort. All requests use a fixed 4,096-token ignore_eos window including a post-answer tail; these are controlled measurements, not completed-answer throughput.

| Cold prefill, C1 | v0.4.1 mean / median prompt tok/s | v0.4.2 mean / median prompt tok/s | Mean / median change |
|---|---:|---:|---:|
| 8k, 5 requests per version | 5,616.6 / 5,629.8 | 5,647.7 / 5,659.0 | +0.55% / +0.52% |
| 32k, 5 requests per version | 6,344.2 / 6,338.5 | 6,352.0 / 6,359.6 | +0.12% / +0.33% |
| 64k, 5 requests per version | 6,377.1 / 6,371.1 | 6,361.1 / 6,345.1 | -0.25% / -0.41% |
| 128k, 5 requests per version | 6,359.6 / 6,336.7 | 6,346.7 / 6,319.5 | -0.20% / -0.27% |

Compared with v0.4.1, mean output rates changed by +8.08% at C1, -1.95% at C2, -0.69% at C3 and +0.15% at C4. Mean target forward rates were 0.23%-0.79% lower; mean and median cold-prefill rates differed by at most 0.56%. Output-rate changes also reflect speculative acceptance in the fixed window, including its post-answer tail. Five sequential repetitions within one startup per image do not establish an isolated speedup or statistical significance.

### Correctness coverage

The measured validation build passed CPU and GPU checks for cache transfer and compressed index ownership, sparse attention and masked reads, CUDA-graph buffer lifetimes, KDA, MHC, MoE routing, vision offload and request scheduling. HiCache coverage included mixed FP8-target/BF16-draft storage and byte/state round trips under Compute Sanitizer, with zero reported sanitizer errors. Model-level checks covered off-grid chunked-prefill checkpoints and five forced host restores followed by device replays in the mixed-storage configuration. These checks exercise the changed paths and carried integrations on the stated two-GPU platform.

### Answer quality and startup

The launcher uses adaptive three/five-step MTP for serving. The following quality and reliability checks used that serving configuration; published performance comparisons use the fixed-MTP standard above.

The adaptive-serving GSM8K run completed all 1,319 requests without request errors. With the unchanged 16,384-token budget and default reasoning, the GLM-aware extractor matched 1,279 answers; 1,277 matches came from completed responses. Two responses exhausted the budget while reasoning and did not produce final answers. The stock extractor matched 1,166 answers. The matched v0.4.1 run had 1,282 GLM-aware matches, 1,172 stock matches and no capped responses. These single runs do not establish quality improvement or equivalence.

The unchanged long-context GSM8K workload scored 144/150 at approximately 73k tokens and 143/150 at approximately 400k tokens, compared with 144/150 and 145/150 for v0.4.1. These checks used a 2,048-token budget and `reasoning_effort: low`; the harness retained extracted answers but not finish reasons, so completion status is unknown.

On the first boot of the measured validation build, the first short thinking response finished in 0.768 seconds and the first sampled response in 0.574 seconds. A 7680×4320 image followed by four simultaneous independent cold 4k-token requests completed without an out-of-memory failure or restart. A 523,787-token prompt also ran with a deliberate 256-token output window. The 524,288-token device pool is shared across requests; it is not a per-request reservation.

The reusable workloads and measurement tools are in [bench/](bench/). Other GPU pairs and tensor-parallel sizes have not been measured.
