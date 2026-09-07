# Measured results

## v0.3.2 measurements, 2026-09-06

Measurements used a v0.3.2 validation build on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe. The GHCR image was built separately from the same pinned inputs and was not separately benchmarked. The configuration used W4A16 experts, FP8 KV, a 450,560-token pool, four running requests, 4,096-token prefill chunks, 28 recurrent-state slots and 32 GB of HiCache per rank.

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

These controlled probes describe this configuration and do not isolate individual optimizations or establish statistical significance.

### Answer quality and startup

Across all 1,319 GSM8K questions, 1,279 answers (97.0%) were scored correct by the GLM-aware answer extractor. The stock AIPerf extractor scored 1,171 (88.8%) on the same responses. All requests succeeded; one response reached the 16,384-token output limit and remains included. Results depend on the stated answer extraction method.

The 150-question long-context controls scored 142 at about 73k tokens and 144 at about 400k. Seven retrieval markers were preserved in order at about 408k tokens through cold processing, device-cache reuse and host-cache restoration. Generated wording can vary across cache paths; byte-identical responses are not guaranteed.

The first sampled chat completed in 0.690 seconds with a 0.470-second time to first token. A 7680x4320 image followed by four independent cold text requests completed without a server restart. Other GPU pairs and tensor-parallel sizes have not been measured.

The reusable workload definitions and measurement tools are in [bench/](bench/). Earlier releases and their supported configurations are described in the [changelog](CHANGELOG.md).
