# Running GLM-5.3-Flash

Download the checkpoint as described in [README.md](README.md), then start the published image:

```bash
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export IMAGE=sglang-glm53-flash-sm120:v0.4.2-rc.2
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v81
./examples/serve-glm53-flash.sh
```

The launcher uses TP2, vision, native adaptive MTP, packed FP8 KV, a 524,288-token pool, four running requests, and 28 BF16 recurrent-state slots. HiCache is optional; set `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` for 32 GB per rank. Use a separate cache directory for each image version.

Prefill uses breakable CUDA graphs for single-request 64- and 128-token tails. Longer prefills retain 4,096-token chunks.

## Reproduce the performance configuration

The [published performance panel](BENCHMARKS.md) uses fixed native MTP. Enable HiCache at 32 GB per rank and use the launcher's command with these MTP arguments:

```text
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

Remove `--speculative-adaptive` and `--speculative-adaptive-config` together with its path argument for the benchmark. Keep the checkpoint, reasoning settings, token pool, concurrency and other launch flags unchanged for a matched comparison. The launcher defaults remain adaptive at five steps/top-k one/six verification tokens with its three/five-step ladder for serving. The benchmark settings do not reduce the response or reasoning budget.

Run the `glm-qualification` panel described in [bench/aiperf/README.md](bench/aiperf/README.md). It measures C1-C4 and cold prefill, reporting output tokens/s after MTP alongside target forward passes/s.

## Build from source

```bash
docker build -f Containerfile -t sglang-glm53-flash-sm120:v0.4.1 .
IMAGE=sglang-glm53-flash-sm120:v0.4.1 ./examples/serve-glm53-flash.sh
```
