# Running `v0.3.2`

This release uses SGLang main `28457f0dca`, including merged GLM support, and FlashInfer main `6c14bbd5ff`, plus the recorded integration patches. The exact internal image completed full qualification; each registry's candidate was promoted digest-identically. Use cache schema `v69` for this build.

```bash
export MODEL_REPO=ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export MODEL_DIR=/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
mkdir -p "$MODEL_DIR"
HF_XET_HIGH_PERFORMANCE=1 hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"

export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.3.2
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v69
./examples/serve-glm53-flash.sh
```

The launcher retains the TP2 profile's settings: vision, adaptive MTP
5/1/6, FlashInfer SM120 DSA with packed FP8 KV, a 450,560-token shared pool,
four running requests, 28 BF16 recurrent-state slots, and decode CUDA graphs
at batch sizes 1 through 4. HiCache is opt-in. Set `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` to match the
qualification profile, which covered host restoration and device reuse.

For first boot only, reduce pool, concurrency, recurrent slots, and graph size
together:

```bash
MAX_TOTAL_TOKENS=131072 \
MAX_RUNNING_REQUESTS=1 \
MAX_MAMBA_CACHE_SIZE=5 \
CUDA_GRAPH_MAX_BS=1 \
./examples/serve-glm53-flash.sh
```

Use a new cache directory for every image/runtime/graph combination. The
qualified internal candidate cache may be retained for its digest-identical
stable promotion; other images require a separate cache.
