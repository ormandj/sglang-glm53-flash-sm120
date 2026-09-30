# Measured results

## v0.5.0: TP4 fixed native MTP

Measurements used a v0.5.0 validation build on four RTX PRO 6000 Blackwell Max-Q 96 GB GPUs (SM120) at 250 W, with PCIe Gen4 ×16 links, working GPU peer access across CPU roots and no NVLink. The GHCR image is built separately from the same pinned inputs and was not separately benchmarked. Source inputs are pinned in [this release's stack lock](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.5.0/stack.lock.json).

The checkpoint was `nvidia/GLM-5.3-Flash-NVFP4`, revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`, at TP4/EP1 with BF16 native MTP, FP8 E4M3 KV, a 1,048,576-token context limit, 2,621,440 shared device tokens, 32-request admission, 224 BF16 recurrent-state slots, 8,192-token prefill chunks and 40 GB of HiCache per rank. Only one request ran during each measured decode window.

Fixed MTP used three draft steps, top-k one and four verification tokens, with adaptive switching disabled. Reasoning and the default chat grammar remained enabled. Each request used the same 16,396-token coding prompt and reached a deliberate 4,096-token output cap with `ignore_eos`.

| Workload | Repetitions | Mean output tok/s after MTP | Median output tok/s after MTP | Mean target forwards/s | Median target forwards/s | Output tok/forward, mean / median |
|---|---:|---:|---:|---:|---:|---:|
| Chat decode C1 | 3 | 197.24 | 197.40 | 75.73 | 76.51 | 2.604 / 2.580 |

Rates are least-squares slopes of the server's cumulative emitted-token and target-forward counters over the steady decode interval with context between 17,408 and 20,480 tokens. Accepted windows lasted 14.57–15.27 seconds. Each window had exactly one request, fixed three-step/four-token verification, no prefill activity and no overlapping benchmark or review traffic. Three chat repetitions were interleaved with separate native-generation controls within one server startup; those controls are not part of the table.

Output tokens include reasoning and a post-answer tail. The tail can increase speculative acceptance, so these are output-capped decode measurements, not naturally completed-answer throughput. Mean and median are taken over the three repetitions; output tokens per forward is the per-repetition output slope divided by the forward slope. These controlled measurements do not necessarily represent real-world performance. There is no matched fixed-MTP comparison against the previous image on this TP4 checkpoint.

Cold prefill throughput was not measured with this fixed-MTP configuration. **TP2 performance was not measured for v0.5.0**, and previous TP2 numbers are not presented as results for this release. Historical results remain in the immutable [v0.4.3 benchmarks](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.3/BENCHMARKS.md).

## Correctness and capacity

The TP4 validation build passed focused CPU and GPU regressions for blocked KDA checkpoint indexing. Eight GPU regression cases passed after the correction and failed against the pre-fix implementation. Five increasing cached-prefix serving stages passed at both 4,096- and 8,192-token prefills.

The adaptive serving profile separately processed a 1,025,012-token prompt with seven ordered recall markers on cold and cached paths, and a forced host-cache restore of 408,064 tokens. Multiple-image capacity checks used four and eight 7680×4320 solid-color images, including four concurrent requests with four images each, followed by 32 concurrent cold text requests without an out-of-memory failure or restart. The processor resizes images according to its token budget; these fixtures test capacity and execution, not OCR or photographic accuracy.

The shared device pool is not a per-request reservation. Image tokens, text tokens and other live requests consume that shared capacity. The supplied adaptive launcher uses the three/five-step ladder for serving; the performance table above uses fixed MTP.
