# Running GLM-5.3-Flash

Download the checkpoint for the desired GPU count using [README.md](README.md). Both profiles use `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0`.

## TP4: NVIDIA NVFP4 checkpoint

```bash
export MODEL_DIR=/srv/models/GLM-5.3-Flash-NVFP4
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v88-tp4
./examples/serve-glm53-flash-tp4.sh
```

Use `nvidia/GLM-5.3-Flash-NVFP4` at revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`. The profile uses TP4/EP1, `modelopt_fp4`, a 1,048,576-token per-request limit, 2,621,440 shared device tokens, 32 running requests, 8,192-token prefills and 224 BF16 recurrent-state slots. The FP8 KV pool and recurrent-state pools are separate allocations.

NVIDIA stores the native MTP layer in BF16 without the ignore declaration used by the loader. The launcher supplies both required settings:

```text
--json-model-override-args '{"text_config":{"quantization_config":{"ignore":["model.layers.45.*"]}}}'
--speculative-draft-model-quantization unquant
```

HiCache defaults to 40 decimal GB per rank, or 160 GB across four ranks, in addition to model-loading and serving memory. Set `ENABLE_HICACHE=0` to disable it. NCCL peer access defaults to `NCCL_P2P_LEVEL=SYS`; the measured PCIe platform supports peer access across CPU roots. Verify P2P support on a different platform before adopting that setting.

## TP2: previous W4A16 checkpoint

```bash
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v88-tp2
./examples/serve-glm53-flash.sh
```

Use `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`, with W4A16 experts and mixed FP8 attention output projections, rather than NVIDIA's checkpoint. This launcher uses TP2/EP1, `modelopt_mixed`, a 524,288-token context limit and shared device pool, four running requests, 4,096-token prefills and 28 BF16 recurrent-state slots. HiCache is optional: `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` allocates 32 GB per rank. [TP2 fixed-MTP measurements](BENCHMARKS.md#v050-tp2-w4a16-fixed-native-mtp) cover C1/C2/C4 decode and cold prefill.

## Common settings

Use separate persistent cache directories for the profiles and a fresh directory when changing image versions. Both launchers keep native FlashInfer NoPE attention, FP8 E4M3 KV, the Triton linear-attention backend, CPU torchvision preprocessing, GLM reasoning/tool parsers and native adaptive MTP. They do not enable expert parallelism beyond EP1.

Breakable prefill graphs cover single-request 64- and 128-token tails. Larger prefills use the profile's 4,096- or 8,192-token chunks, with a 2,048-token internal KDA block size. Full decode graphs cover batches 1 through 4 at TP2 and 1, 2, 3, 4, 8, 16 and 32 at TP4. Recurrent-state slots are not KV tokens; keep at least five slots per running request.

Adaptive MTP defaults to five draft steps, top-k one and six verification tokens. Keep the bundled `examples/glm53-adaptive.json` mounted: it limits the adaptive ladder to three and five steps and avoids sizing speculative state for an unnecessarily large ladder at long context.

## Reproduce the fixed-MTP measurement

The [v0.5.0 measurements](BENCHMARKS.md) use the TP4 NVIDIA and TP2 W4A16 profiles with their documented settings and these MTP arguments:

```text
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

Remove `--speculative-adaptive` and `--speculative-adaptive-config` together with its path argument. Keep reasoning enabled and the default chat grammar. Measure TP4 C1 or TP2 C1/C2/C4 with 16K coding prompts including chat framing and a deliberate 4,096-token output cap using `ignore_eos`; derive target forward and output rates over the steady decode window described in BENCHMARKS.md. TP2 cold prefill uses C1 and five requests per 8K/32K/64K/128K shape, ending at the first output token.

## Build from source

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
docker build -f Containerfile \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-glm53-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.5.0-rc.4 .
CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v88-tp4 \
  IMAGE=sglang-glm53-flash-sm120:v0.5.0-rc.4 ./examples/serve-glm53-flash-tp4.sh
```

For a local TP2 build, use the previous W4A16 checkpoint and `./examples/serve-glm53-flash.sh` with a separate TP2 cache directory.

[BUILDING.md](BUILDING.md) describes the CI workflows and the separate steps required to publish and verify a GitHub Release. A local build or successful candidate-image workflow does not create a repository Release.
