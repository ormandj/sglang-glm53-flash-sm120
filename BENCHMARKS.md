# Measured results

## v0.6.1: TP4 FP8 fixed native MTP

Measured on 2026-10-11 on four RTX PRO 6000 Blackwell Max-Q 96 GB GPUs (SM120) at 250 W, with PCIe Gen4 ×16 links, working GPU peer access across CPU roots and no NVLink. The checkpoint was `zai-org/GLM-5.3-Flash`, revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. Measurements used the internal v0.6.1 release build (candidate `v0.6.1-rc.1`, promoted unchanged). The GHCR image is built separately from the same pinned inputs and was not separately benchmarked.

TP4/EP1 used the checkpoint's FP8 block-scaled weights, FP8 E4M3 KV, a 1,048,576-token context limit and shared device pool, 16-request admission, 96 BF16 recurrent-state slots, 8,192-token prefill chunks and 40 GB of HiCache per rank, as in the supplied launcher. Native MTP used three draft steps, top-k one and four verification tokens, with the draft LM head in FP8 and the target LM head in BF16.

| Decode concurrency | Repetitions | Mean aggregate output tok/s after MTP | Median aggregate output tok/s after MTP | Mean target forwards/s | Median target forwards/s | Output tok/forward/request, mean / median |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 5 | 230.50 | 210.97 | 78.24 | 78.18 | 2.966 / 2.752 |
| C2 | 5 | 312.74 | 309.81 | 54.86 | 54.73 | 2.868 / 2.867 |
| C4 | 5 | 433.77 | 423.11 | 37.97 | 38.05 | 2.857 / 2.769 |

The workload used 16K coding prompts with chat framing and a forced 4,096-token output cap (`ignore_eos`), including reasoning at the template's default `max` effort and a post-answer tail. Output rates are aggregate across the stated concurrency. Target forwards count batch iterations on rank zero, not one forward per request. Rates use least-squares counter slopes during exact-concurrency decode windows with average context between 17,408 and 20,480 tokens. Output tok/forward/request uses emitted-token and target-forward counter deltas divided by concurrency. The 15 windows lasted 9.67–28.00 seconds with 30–85 samples each and had no prefill and no other traffic; every window retained three-step/four-token verification.

Five prompt-seed repetitions ran per concurrency within one server startup. They are prompt-path samples, not independent deployment replicates. The post-answer tail can increase speculative acceptance; these controlled rates do not necessarily represent naturally completed answers or application throughput.

Cold prefill used five sequential C1 requests per shape, stopped after the first output token. Cache-hit tokens were zero. Per-request prompt tok/s is actual input tokens divided by time to first token; the table reports its mean and median across five requests. Window prompt tok/s is total actual input tokens divided by elapsed time from the first request send to the last first token, including inter-request gaps. Both include serving overhead and are not GPU-compute-only rates.

| Nominal prompt size | Requests | Mean request prompt tok/s | Median request prompt tok/s | Window prompt tok/s | Median TTFT, seconds |
|---|---:|---:|---:|---:|---:|
| 8K | 5 | 10,446.47 | 10,573.97 | 10,418.53 | 0.776 |
| 32K | 5 | 10,213.83 | 10,556.18 | 10,159.56 | 3.105 |
| 64K | 5 | 10,476.89 | 10,499.96 | 10,472.78 | 6.243 |
| 128K | 5 | 10,165.51 | 10,173.82 | 10,162.44 | 12.859 |

Token targets were 8,192, 32,768, 65,536 and 130,816 before chat framing. All cells completed without request errors or server restarts. [Per-repetition and per-request data](bench/results/v0.6.1-tp4-fp8-fixed-mtp.json) includes exact input sizes, windows and settings.

Against [v0.6.0](bench/results/v0.6.0-tp4-fp8-fixed-mtp.json), measured the same way, mean target forwards/s changed from 77.79 to 78.24 at C1, 54.93 to 54.86 at C2 and 37.53 to 37.97 at C4, and cold prefill stayed at 10.1–10.6K prompt tok/s. These differences are within the spread of repeated runs. v0.6.1 changes no numerics: the draft MoE configuration produces bitwise-identical outputs ([kernel measurements](https://github.com/sgl-project/sglang/pull/43647)), and the prefill-overlap changes only add safeguards.

## v0.6.1: TP2 W4A16 fixed native MTP

v0.6.1 changes no code that this profile executes: the TP2 launcher disables the prefill microbatch overlap, and the new draft MoE configuration applies only to the TP4 expert shape. The v0.6.1 image passed TP2 startup and first-traffic checks. The measurements below are the v0.6.0 ones.

Measured on 2026-10-11 on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs (SM120) at 250 W, with PCIe Gen4 ×16 links across CPU roots, working P2P/IPC and no NVLink. The checkpoint was `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`, revision `ee0989a944b0e213589191d7fca63af825a0741e`. Measurements used a v0.6.0 validation build; the GHCR image is built separately from the same pinned inputs and was not separately benchmarked.

TP2/EP1 used `modelopt_mixed`, FP8 E4M3 KV, a 524,288-token context limit and shared device pool, four-request admission, 28 BF16 recurrent-state slots, 4,096-token prefill chunks, no HiCache and the prefill microbatch overlap disabled, as in the supplied launcher. Native MTP used three draft steps, top-k one and four verification tokens, with the draft LM head in FP8 and the target LM head in BF16.

| Decode concurrency | Repetitions | Mean aggregate output tok/s after MTP | Median aggregate output tok/s after MTP | Mean target forwards/s | Median target forwards/s | Output tok/forward/request, mean / median |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 5 | 191.95 | 181.77 | 64.94 | 65.16 | 2.960 / 2.798 |
| C2 | 5 | 283.98 | 278.70 | 48.55 | 48.57 | 2.914 / 2.850 |
| C4 | 5 | 391.07 | 386.16 | 32.79 | 32.58 | 2.990 / 2.973 |

The workload, rate definitions and repetition design match the TP4 section. The 15 windows lasted 12.33–30.34 seconds with 38–92 samples each, with no prefill and no other traffic; every window retained three-step/four-token verification.

Against the matched v0.5.0 TP2 measurement (same checkpoint, GPU model, power limit, runtime limits, workload and fixed MTP; the v0.5.0 launcher set `SGLANG_LM_HEAD_FP8=1`, which that image did not implement, so both LM heads ran in BF16), mean target forwards/s rose from 61.78 to 64.94 at C1 (+5.1%), 47.23 to 48.55 at C2 (+2.8%) and 32.29 to 32.79 at C4 (+1.5%). Output tok/s also depends on per-seed speculative acceptance, so forward rates are the comparison measure.

Cold prefill followed the TP4 procedure.

| Nominal prompt size | Requests | Mean request prompt tok/s | Median request prompt tok/s | Window prompt tok/s | Median TTFT, seconds |
|---|---:|---:|---:|---:|---:|
| 8K | 5 | 5,673.37 | 5,698.09 | 5,667.36 | 1.440 |
| 32K | 5 | 6,160.28 | 6,162.44 | 6,158.29 | 5.319 |
| 64K | 5 | 6,156.05 | 6,158.10 | 6,154.77 | 10.644 |
| 128K | 5 | 6,086.05 | 6,076.06 | 6,084.99 | 21.532 |

Window prompt tok/s at 8K, 32K, 64K and 128K was 5,135, 6,074, 6,123 and 6,131 on v0.5.0. Token targets were 8,192, 32,768, 65,536 and 130,816 before chat framing. All cells completed without request errors or server restarts. [Per-repetition and per-request data](bench/results/v0.6.0-tp2-w4a16-fixed-mtp.json) includes exact input sizes, windows and settings.

## Correctness and capacity

Both profiles completed the 1,319 GSM8K test questions at temperature zero with the default `max` reasoning effort and a 16,384-token output cap, without request errors: 1,279 correct (97.0%) at TP4 on v0.6.1, as on v0.6.0, and 1,280 (97.0%) at TP2 on v0.6.0, scored on the final numeric answer extracted from each response.

At TP4 on v0.6.1, a 1,025,012-token prompt with seven ordered recall markers returned all seven in 107.8 seconds (107.9 on v0.6.0). On v0.6.0, sixteen concurrent 60K-token cold requests completed without errors, and a 200K-token prefix evicted to the host cache was restored with a 0.19-second time to first token. v0.6.1 leaves 3.5 GB per GPU free after graph capture, unchanged from v0.6.0. Startup checks on both profiles covered a 7680×4320 image followed by a full cohort of cold text requests, multiple-image requests and cached-prefix reuse across increasing prefixes, without an out-of-memory failure or restart. The processor resizes images according to its token budget; these fixtures test capacity and execution, not OCR or photographic accuracy.

The shared device pool is not a per-request reservation. Image tokens, text tokens and other live requests consume that shared capacity.

Historical results remain in the immutable [v0.5.0 benchmarks](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.5.0/BENCHMARKS.md).
