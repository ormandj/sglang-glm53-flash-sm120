# Changelog

## v0.4.0 (stable; 2026-09-08)

- Increase the context limit and shared device token pool to 524,288 tokens while retaining four-request admission, the same checkpoint and quantization, native speculative decoding, reasoning and vision.
- Store native GLM FP8 cache rows in 528 bytes instead of 656 without requantizing values. Reuse host sequence lengths in KDA and use breakable CUDA graphs for small prefill tails.
- Preserve valid sparse-attention tails after missing entries, prevent masked attention from reading mutable padding slots, and align compressed DSA cache ownership in both standalone and hybrid pools.
- Correct graph-buffer lifetime and native attention workspace bounds, and warm up cached-prefix and sampled paths before readiness. Refresh SGLang and FlashInfer source inputs.
- Use a fresh cache directory for this version: `/srv/cache/sglang-glm53-flash-sm120-v78`.

### Measurements

Measured on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe, with 32 GB of HiCache per GPU and fixed native MTP at three draft steps, top-k one and four verification tokens. Fixed MTP is the performance-comparison standard; the launcher uses adaptive MTP for serving. These measurements use a v0.4.0 validation build; the GHCR image was built separately from the same pinned inputs and was not separately benchmarked.

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

Decode tok/s is aggregate output after MTP, including reasoning, over the selected steady portion of a fixed 4,096-token response window. Its post-answer tail can increase speculative acceptance. Forward passes/s counts target-model iterations. Prefill values summarize each cold request's prompt tokens divided by time to first token. These controlled measurements do not necessarily represent real-world performance. See [BENCHMARKS.md](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.0/BENCHMARKS.md) for methodology, fixed-MTP comparisons and quality results.

### Known limitations

- Requesting input logprobs across a long prompt can exhaust GPU memory and restart the server. Score continuations with `logprob_start_len` at the prompt boundary.
- Keep `--max-prefill-tokens` and `--chunked-prefill-size` at 4096 with the supplied memory configuration. Four running requests share the device token pool; the context limit is not separate capacity for each concurrent request.
- Other GPU pairs and tensor-parallel sizes have not been tested.

Image: `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.4.0`.

Digest: `sha256:d4649b9b957942407d49885a0cd42589b25f2ab53c54c37694c8a2f43473d7a9`.

## v0.3.2 (stable; 2026-09-06)

- Update SGLang to the merged GLM-5.3-Flash implementation and its current interfaces. Retain the attention-metadata optimization and remove superseded experimental paths.
- Retain the W4A16 expert optimizations and memory-lifetime corrections from v0.3.1. The checkpoint, FP8 KV cache, vision input, adaptive speculative decoding, 450,560-token context and four-request limit remain supported.
- Use a fresh kernel cache directory for this version: `/srv/cache/sglang-glm53-flash-sm120-v69`.

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
