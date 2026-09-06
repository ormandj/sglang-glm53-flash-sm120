# Running `v0.3.1-rc.1`

This candidate is built internally and has completed the route-prefix experiment. It is not release-qualified; the qualified internal serving release remains `v0.3.0`. See the primary project evidence for the concurrent gains and unresolved single-request decrease.

```bash
export MODEL_REPO=ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export MODEL_DIR=/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
mkdir -p "$MODEL_DIR"
HF_XET_HIGH_PERFORMANCE=1 hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"

export IMAGE=sglang-glm53-flash-sm120:v0.3.1-rc.1
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v68
./examples/serve-glm53-flash.sh
```

This candidate is not yet built or qualified. The launcher retains the TP2
profile's settings: vision, adaptive MTP
5/1/6, FlashInfer SM120 DSA with packed FP8 KV, a 450,560-token shared pool,
four running requests, 28 BF16 recurrent-state slots, and decode CUDA graphs
at batch sizes 1 through 4. HiCache is opt-in (`ENABLE_HICACHE=1`); the
exact-image qualification must cover both host restoration and device reuse.

For first boot only, reduce pool, concurrency, recurrent slots, and graph size
together:

```bash
MAX_TOTAL_TOKENS=131072 \
MAX_RUNNING_REQUESTS=1 \
MAX_MAMBA_CACHE_SIZE=5 \
CUDA_GRAPH_MAX_BS=1 \
./examples/serve-glm53-flash.sh
```

Use a new cache directory for every candidate/runtime/graph combination. Never
reuse an earlier candidate's cache with `v0.3.1-rc.1`.
