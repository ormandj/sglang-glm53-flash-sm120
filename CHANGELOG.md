# Changelog

## v0.3.2 (stable; 2026-09-06)

- Update SGLang to the merged GLM-5.3-Flash implementation and its current interfaces. Retain the attention-metadata optimization and remove superseded experimental paths.
- Retain the W4A16 expert optimizations and memory-lifetime corrections from v0.3.1. The checkpoint, FP8 KV cache, vision input, adaptive speculative decoding, 450,560-token context and four-request limit remain supported.
- Use a fresh kernel cache directory for this version: `/srv/cache/sglang-glm53-flash-sm120-v69`.

### Measurements

Measured on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe, with 32 GB of HiCache per GPU. These measurements use the v0.3.2 validation build; the GHCR image was built separately from the same pinned inputs and was not separately benchmarked.

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

Decode tok/s is aggregate output after MTP, including reasoning, over a fixed 4,096-token response window. Its post-answer tail can increase speculative acceptance. Forward passes/s counts target-model iterations. Prefill values summarize each cold request's prompt tokens divided by time to first token. These controlled measurements do not necessarily represent real-world performance. See [BENCHMARKS.md](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/main/BENCHMARKS.md#v032-measurements-2026-09-06) for latency and supplemental decode measurements.

### Known limitations

- Requesting input logprobs across a long prompt can exhaust GPU memory and restart the server. When scoring a continuation, set `logprob_start_len` to the prompt boundary.
- Keep `--max-prefill-tokens` and `--chunked-prefill-size` at 4096 with the supplied memory configuration. Larger batches require additional GPU memory.
- Other GPU pairs and tensor-parallel sizes have not been tested.

Image: `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.3.2`.

Digest: `sha256:5f12516c84abb3a74f135ba43a18021b05c6ca14c186a6583087aefc245593fb`.

## v0.3.1 (stable; 2026-09-06)

- Reduce routing-metadata work for small batches of W4A16 experts during concurrent decoding.
- Correct restoration of target and speculative-draft attention indexes from the host cache, addressing the long-prefix HiCache corruption found in v0.2.1.
- Retain memory-lifetime and request-lifecycle corrections, CPU image preprocessing and sampled warmup before the server becomes ready.
- Keep the 450,560-token context and four-request limit. Use a fresh kernel cache directory with suffix `v68`.

## v0.2.1 (stable; 2026-09-04)

- Correct sparse-attention selection when score bins fill or overflow, including large ties and signed-zero cases.
- Supply image embeddings to speculative decoding on the first multimodal request. Correct tool-result media ordering and video-worker cleanup.
- Preserve request-admission limits when recurrent-state slots are exhausted.
- Known issue found after release: restoring long prefixes from HiCache can corrupt output. Keep HiCache disabled on v0.2.1 or upgrade to v0.3.1 or later.

## v0.2.0 (stable; 2026-09-03)

- Enable PCIe IPC all-reduce for small tensor-parallel transfers.
- Fuse KDA gate projections under the quantized configuration and use device-side attention metadata during speculative decoding.
- Use an FP8 output projection shared by the target model and speculative drafter.
- Retain the context capacity and recurrent-state pool configuration.

## v0.1.4 (stable; 2026-09-02)

- Update SGLang and FlashInfer with the current GLM model support and sparse-attention kernels.
- Retain the 450,560-token context and 28 recurrent-state slots. Use a fresh kernel cache directory with suffix `v56`.

## v0.1.3 (stable; 2026-09-02)

- Reduce the context and token pool from 499,712 to 450,560 to leave memory for image encoding while long requests are active.
- Skip optional recurrent-state checkpoints when the pool is exhausted, preventing a scheduler assertion during concurrent cold prefills.

## v0.1.2 (stable; 2026-09-02)

- Recheck request admission during decoding so a fourth request can start when capacity is available, instead of waiting for one of the first three to finish.

## v0.1.1 (stable; 2026-09-02)

- Fix a sparse-attention top-k memory fault reported by @bold84 and @sousekd, with the fix contributed by @bold84.
- Release raw image tensors after encoding and bound KDA prefill workspace, allowing large images within the model's 8,000-token image budget.
- Keep image preprocessing on the CPU to avoid creating another CUDA context.
- Warm up sampled generation and vision operations before the server becomes ready.
- Increase the recurrent-state pool to 28 slots and correct packed attention-cache transfers.

## v0.1.0 (stable; 2026-09-01)

- Initial stable container for GLM-5.3-Flash on two SM120 GPUs, with W4A16 experts, sparse attention, vision and speculative decoding.
- Include and mount the adaptive-speculation configuration in the launcher so speculative state buffers are sized for the configured draft widths.
