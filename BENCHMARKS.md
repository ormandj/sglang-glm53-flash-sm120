# Measured results

## v0.5.0: TP4 fixed native MTP

Measurements used a v0.5.0 validation build on four RTX PRO 6000 Blackwell Max-Q 96 GB GPUs (SM120) at 250 W, with PCIe Gen4 ×16 links, working GPU peer access across CPU roots and no NVLink. The GHCR image is built separately from the same pinned inputs and was not separately benchmarked. Source inputs are pinned in [this release's stack lock](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.5.0/stack.lock.json).

The checkpoint was `nvidia/GLM-5.3-Flash-NVFP4`, revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`, at TP4/EP1 with BF16 native MTP, FP8 E4M3 KV, a 1,048,576-token context limit, 2,621,440 shared device tokens, 32-request admission, 224 BF16 recurrent-state slots, 8,192-token prefill chunks and 40 GB of HiCache per rank. Only one request ran during each measured decode window.

Fixed MTP used three draft steps, top-k one and four verification tokens, with adaptive switching disabled. Reasoning used the template’s default `max` effort, and the default chat grammar remained enabled. Each request used the same 16,396-token coding prompt and reached a deliberate 4,096-token output cap with `ignore_eos`.

| Workload | Repetitions | Mean output tok/s after MTP | Median output tok/s after MTP | Mean target forwards/s | Median target forwards/s | Output tok/forward, mean / median |
|---|---:|---:|---:|---:|---:|---:|
| Chat decode C1 | 3 | 197.24 | 197.40 | 75.73 | 76.51 | 2.604 / 2.580 |

Rates are least-squares slopes of the server's cumulative emitted-token and target-forward counters over the steady decode interval with context between 17,408 and 20,480 tokens. Accepted windows lasted 14.57–15.27 seconds. Each window had exactly one request, fixed three-step/four-token verification, no prefill activity and no overlapping benchmark or review traffic. Three chat repetitions were interleaved with separate native-generation controls within one server startup; those controls are not part of the table.

Output tokens include reasoning and a post-answer tail. The tail can increase speculative acceptance, so these are output-capped decode measurements, not naturally completed-answer throughput. Mean and median are taken over the three repetitions; output tokens per forward is the per-repetition output slope divided by the forward slope. These controlled measurements do not necessarily represent real-world performance. There is no matched fixed-MTP comparison against the previous image on this TP4 checkpoint.

Cold prefill throughput was not measured for this TP4 configuration. Historical results remain in the immutable [v0.4.3 benchmarks](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.3/BENCHMARKS.md).

## v0.5.0: TP2 W4A16 fixed native MTP

Measured on 2026-09-30 using a v0.5.0 validation build on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs (SM120) at 250 W, with PCIe Gen4 ×16 links across CPU roots, working P2P/IPC and no NVLink. The separately built GHCR image was not benchmarked. The checkpoint was `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`, revision `ee0989a944b0e213589191d7fca63af825a0741e`.

TP2/EP1 used `modelopt_mixed`, FP8 E4M3 KV, a 524,288-token context limit and shared device pool, four-request admission, 28 BF16 recurrent-state slots, 4,096-token prefill chunks and no HiCache. Measurements used fixed native MTP with three draft steps, top-k one and four verification tokens, with adaptive switching disabled. The supplied TP2 launcher instead defaults to adaptive five-step/six-token verification; these results measure the fixed configuration.

| Decode concurrency | Repetitions | Mean aggregate output tok/s after MTP | Median aggregate output tok/s after MTP | Mean target forwards/s | Median target forwards/s | Output tok/forward/request, mean / median |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 5 | 181.60 | 167.02 | 61.78 | 61.76 | 2.950 / 2.680 |
| C2 | 5 | 287.55 | 284.11 | 47.23 | 47.25 | 3.043 / 3.014 |
| C4 | 5 | 381.50 | 376.37 | 32.29 | 32.15 | 2.952 / 2.926 |

The workload used 16K coding prompts with chat framing and a forced 4,096-token output cap (`ignore_eos`), including reasoning at the template's default `max` effort and a post-answer tail. Output rates are aggregate across the stated concurrency. Target forwards count batch iterations on rank zero, not one forward per request. Rates use least-squares counter slopes during exact-concurrency decode windows with average context between 17,408 and 20,480 tokens. Output tok/forward/request uses emitted-token and target-forward counter deltas divided by concurrency. Windows lasted 13.67–30.68 seconds with 42–93 samples each, with no prefill, other inference, review or build activity. All 15 windows retained fixed three-step/four-token verification.

Five prompt-seed repetitions ran per concurrency within one server startup. They are prompt-path samples, not independent deployment replicates. The post-answer tail can increase speculative acceptance; these controlled rates do not necessarily represent naturally completed answers or application throughput. TP2 and TP4 use different checkpoints and settings, so their tables are not a matched scaling comparison.

Cold prefill used five sequential C1 requests per shape, stopped after the first output token. Cache-hit tokens were zero. Per-request prompt tok/s is actual input tokens divided by time to first token; the table reports its mean and median across five requests. Window prompt tok/s is total actual input tokens divided by elapsed time from the first request send to the last first token, including inter-request gaps. Both include serving overhead and are not GPU-compute-only rates.

| Nominal prompt size | Requests | Mean request prompt tok/s | Median request prompt tok/s | Window prompt tok/s | Median TTFT, seconds |
|---|---:|---:|---:|---:|---:|
| 8K | 5 | 5,190.98 | 5,415.84 | 5,135.45 | 1.515 |
| 32K | 5 | 6,076.75 | 6,100.63 | 6,073.76 | 5.373 |
| 64K | 5 | 6,124.89 | 6,133.47 | 6,122.75 | 10.687 |
| 128K | 5 | 6,132.49 | 6,129.46 | 6,131.15 | 21.344 |

Token targets were 8,192, 32,768, 65,536 and 130,816 before chat framing. All cells completed without request errors or server restarts. [Per-repetition and per-request data](bench/results/v0.5.0-tp2-fixed-mtp.json) includes exact input sizes, windows, settings and client revision.

## Correctness and capacity

The TP4 validation build passed focused CPU and GPU regressions for blocked KDA checkpoint indexing. Eight GPU regression cases passed after the correction and failed against the pre-fix implementation. Five increasing cached-prefix serving stages passed at both 4,096- and 8,192-token prefills.

The adaptive serving profile separately processed a 1,025,012-token prompt with seven ordered recall markers on cold and cached paths, and a forced host-cache restore of 408,064 tokens. Multiple-image capacity checks used four and eight 7680×4320 solid-color images, including four concurrent requests with four images each, followed by 32 concurrent cold text requests without an out-of-memory failure or restart. The processor resizes images according to its token budget; these fixtures test capacity and execution, not OCR or photographic accuracy.

The shared device pool is not a per-request reservation. Image tokens, text tokens and other live requests consume that shared capacity. The supplied adaptive launcher uses the three/five-step ladder for serving; the performance table above uses fixed MTP.
