# Running GLM-5.3-Flash

Download the checkpoint for the desired GPU count using [README.md](README.md). Both profiles use `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.6.1`.

## TP4: Z.ai FP8 checkpoint

```bash
export MODEL_DIR=/srv/models/GLM-5.3-Flash
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.6.1
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v90-tp4
./examples/serve-glm53-flash-tp4.sh
```

Use `zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. Its experts, shared and dense MLPs and sparse-attention projections are FP8 E4M3 with 128×128 block scales; the linear-attention projections, indexer, router, embeddings and LM head stay BF16. The profile uses TP4/EP1, a 1,048,576-token per-request limit and shared device pool, 16 running requests, 8,192-token prefills and 96 BF16 recurrent-state slots. The FP8 KV pool and recurrent-state pools are separate allocations.

HiCache defaults to 40 decimal GB per rank, or 160 GB across four ranks, in addition to model-loading and serving memory. Set `ENABLE_HICACHE=0` to disable it. NCCL peer access defaults to `NCCL_P2P_LEVEL=SYS`; the measured PCIe platform supports peer access across CPU roots. Verify P2P support on a different platform before adopting that setting.

## TP2: W4A16 checkpoint

```bash
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.6.1
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v90-tp2
./examples/serve-glm53-flash.sh
```

Use `ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO` at revision `ee0989a944b0e213589191d7fca63af825a0741e`, with W4A16 NVFP4 routed experts and FP8 weight-only projections. This launcher uses TP2/EP1, `modelopt_mixed`, a 524,288-token context limit and shared device pool, four running requests, 4,096-token prefills and 28 BF16 recurrent-state slots. HiCache is optional: `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` allocates 32 GB per rank.

## Common settings

Use separate persistent cache directories for the profiles and a fresh directory when changing image versions. Both launchers keep native FlashInfer NoPE attention, FP8 E4M3 KV, the Triton linear-attention backend, CPU torchvision preprocessing and the GLM reasoning/tool parsers. They do not enable expert parallelism beyond EP1.

Both launchers use fixed native MTP:

```text
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

`SGLANG_DRAFT_LM_HEAD_FP8=1` runs only the draft model's vocabulary projection as an FP8 GEMM. The target model's LM head stays BF16, so generated tokens are those the target model verifies. The shared FP8 head (`SGLANG_LM_HEAD_FP8=1`) also quantizes the target head and changes output logits; neither launcher enables it.

Breakable prefill graphs cover single-request tails up to 128 tokens at TP2 and up to 512 tokens at TP4. Larger prefills use the profile's 4,096- or 8,192-token chunks, with a 2,048-token internal KDA block size. Full decode graphs cover batches 1 through 4 at TP2 and 1, 2, 3, 4, 8 and 16 at TP4. Recurrent-state slots are not KV tokens; keep at least five slots per running request.

The image enables two prefill optimizations by default. A single-request prefill chunk of at least 4,096 tokens runs as two halves whose tensor-parallel all-reduces overlap the other half's compute. It speeds up TP4 prefill. At TP2 with 4,096-token chunks it measured slower in a same-session comparison (32K and 128K cold prompts: 5.8K prompt tok/s with it, 6.1K without), so the TP2 launcher sets `SGLANG_PREFILL_MBO=0`. At long context, the sparse-attention indexer splits its query rows across GPUs and gathers the selected tokens; set `SGLANG_DSA_INDEXER_ROW_SPLIT_MIN_ELEMS=0` to disable it.

## Reproduce the fixed-MTP measurements

The [v0.6.1 measurements](BENCHMARKS.md) use the supplied launchers. Keep reasoning enabled and the default chat grammar. Measure C1/C2/C4 decode with 16K coding prompts including chat framing and a deliberate 4,096-token output cap using `ignore_eos`; derive target forward and output rates over the steady decode window described in BENCHMARKS.md. Cold prefill uses C1 and five requests per 8K/32K/64K/128K shape, ending at the first output token.

## Build from source

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
docker build -f Containerfile \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-glm53-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.6.1-rc.1 .
CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v90-tp4 \
  IMAGE=sglang-glm53-flash-sm120:v0.6.1-rc.1 ./examples/serve-glm53-flash-tp4.sh
```

For a local TP2 build, use the W4A16 checkpoint and `./examples/serve-glm53-flash.sh` with a separate TP2 cache directory.

[BUILDING.md](BUILDING.md) describes the CI workflows and the separate steps required to publish and verify a GitHub Release. A local build or successful candidate-image workflow does not create a repository Release.
