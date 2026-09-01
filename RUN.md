# Running `v0.1.1-rc.5`

```bash
export MODEL_REPO=ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export MODEL_DIR=/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
mkdir -p "$MODEL_DIR"
HF_XET_HIGH_PERFORMANCE=1 hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"

export IMAGE=sglang-glm53-flash-sm120:v0.1.1-rc.5
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v54
./examples/serve-glm53-flash.sh
```

The default launcher is the intended TP2 qualification envelope: vision,
native adaptive MTP 5/1/6, native FlashInfer SM120 DSA with packed FP8 KV, a
499,712-token shared pool, C4, 28 BF16 recurrent-state slots, and decode CUDA
graphs through batch size four. Those defaults are targets, not measurements.

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
reuse an earlier candidate's cache with `v0.1.1-rc.5`.
