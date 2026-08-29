# Running `v0.1.0-rc.19`

```bash
export IMAGE=sglang-glm53-flash-sm120:v0.1.0-rc.19
export MODEL_DIR=/models/GLM-5.3-Flash-W4A16-E4M3-K32-MSE
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v11
./examples/serve-glm53-flash.sh
```

The model directory must be the locally produced ModelOpt artifact described in
[QUANTIZATION.md](QUANTIZATION.md). Do not point this profile at the deleted
MXFP4 artifact: its metadata and weight contract are different.

The default profile enables vision and native adaptive MTP, uses TP=2/EP=1,
requests a 524,288-token shared pool, admits four live requests, reserves 20
BF16 recurrent-state slots, keeps the protected BF16 shared experts outside the
FP4 fused bank, and captures decode graphs through batch size 4.
Override values explicitly during staged qualification, for example:

```bash
MAX_TOTAL_TOKENS=131072 \
MAX_RUNNING_REQUESTS=1 \
MAX_MAMBA_CACHE_SIZE=5 \
CUDA_GRAPH_MAX_BS=1 \
./examples/serve-glm53-flash.sh
```

That reduced profile is only a bring-up control. The target remains C4 and
roughly 500K shared tokens. After it passes, measure C8 with both
`MAX_RUNNING_REQUESTS=8` and `MAX_MAMBA_CACHE_SIZE=40`; changing only one of
them produces an invalid comparison.

Use a fresh cache directory for every image/runtime/graph combination. The
`v11` suffix is mandatory because SGLang, FlashInfer JIT sources, raw FP8 DSA,
and the W4A16 packer all changed from the old candidate.

## Qualification order

1. Inspect model metadata, tensor counts, shapes, dtypes, byte totals, and
   hashes before allocating GPU memory.
2. Run packed-kernel versus dequantized-reference numerical tests on actual
   expert tensors.
3. Boot target-only at a small pool, no CUDA graphs, and MTP off as an A/B
   diagnostic; this does not qualify the intended profile.
4. Verify deterministic text, tool calls with nested schemas, and a real image.
5. Enable native MTP and measure acceptance plus output equivalence.
6. Raise the shared pool to the maximum that leaves safe per-rank headroom,
   then qualify C4 and test C8 burst behavior.
7. Record prefill/decode throughput, latency, memory pools, and exact image/model
   digests in [BENCHMARKS.md](BENCHMARKS.md).

Do not publish server process arguments in evidence: cluster arguments may
contain an API key. Filter logs to the memory, backend, correctness, and timing
lines needed for the receipt.
