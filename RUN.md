# Running a source build

Build the immutable image from this checkout with `docker build -f Containerfile -t sglang-glm53-flash-sm120:v0.4.0-rc.5 .`.

```bash
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export IMAGE=sglang-glm53-flash-sm120:v0.4.0-rc.5
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v74
./examples/serve-glm53-flash.sh
```

The launcher uses TP2, vision, native adaptive MTP, packed FP8 KV, a 450,560-token pool, four running requests, and 28 BF16 recurrent-state slots. HiCache is optional; set `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` for 32 GB per rank. Use a separate cache directory for each image version.
