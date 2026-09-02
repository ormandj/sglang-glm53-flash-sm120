# SGLang for GLM-5.3-Flash on SM120

A reproducible Linux x86_64 image for serving the published
[`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO)
checkpoint on two NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) GPUs with
SGLang, TP2, no NVLink required.

**Hugging Face model:**
[`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO)

Current release: `v0.1.1-rc.14` for the W4A16 NVFP4 K32 experts + FP8
weight-only checkpoint. Source candidate, not yet built or qualified
(internal build name
`sglang-glm53-flash-sm120:v0.1.1-rc.14`):

```text
ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.1.1-rc.14
```

`v0.1.0` (digest-identical promotion of `v0.1.0-rc.71`) remains the
latest stable, qualified image.

## What you get

Measured results below were qualified on `v0.1.0`. `v0.1.1-rc.14`
changes only the DSA radix top-k kernel (value-correct overflow descent
plus crash-proofed refine rounds) and is not yet built or qualified;
requalification is required before any number below may be cited for it.

- **C4 serving with a 499,712-token KV pool and context limit** — four
  concurrent agentic requests sharing a full ~500k-token budget, with a
  502,784-token single-request cold prefill demonstrated.
- **HiCache host tier**: 13.5M KV tokens (82.9 GB host memory) plus a mamba
  state tier, so evicted long prefixes resume from RAM instead of
  recomputing.
- **Adaptive EAGLE/NextN MTP speculative decoding** (candidate steps [3,5],
  acceptance-driven) plus PCIe IPC allreduce for TP2 small reduces:
  decode plateaus of 169.7 tok/s at C1, 235.7 at C2, and 347.0 sustained
  at C4; 257–267 tok/s at C1 on math where acceptance reaches ~6 tokens.
  Cold 200k prefill 5,153 tok/s.
- **Vision intact**, exercised through the OpenAI-compatible image input.
- **GSM8K 97.2%** (1,282/1,319, zero-shot, temperature 0, position-based
  answer extraction; 89.5% under the raw last-number grader) — identical to
  the BF16-attention control, so the FP8 tier is measured-lossless here.
- **Reproducible end to end**: exact upstream SGLang/FlashInfer commits plus
  checksummed patches (`stack.lock.json`, `scripts/verify-patches.sh`), the
  quantization producers (`quantization/`), the benchmark harness
  (`bench/`), and raw receipts (`evidence/`).

### Performance and capacity at a glance

All rows are the shipped configuration (adaptive MTP [3,5] + PCIe IPC
allreduce), measured with the vendored harness; decode is the n=5 coding
corpus at 4,096 output tokens, temperature 0.

| Decode (n=5 mean OLS plateau over server decode counters) | Rate |
|---|---:|
| C1 (pure equal-context plateau) | 169.7 tok/s |
| C2 (pure equal-context plateau) | 235.7 tok/s |
| C4 (sustained with request refill) | 347.0 tok/s |
| C1 on math-class content (acceptance ~6) | 257–267 tok/s |

| Cold prefill (prompt tokens / TTFT) | Rate |
|---|---:|
| 200k, C1 | 5,153 tok/s |
| 4 x 120k concurrent (aggregate) | 4,985 tok/s |

| Capacity and quality | Result |
|---|---|
| KV pool / context limit | 499,712 tokens, C4 |
| Largest single cold prefill served | 502,784 tokens |
| HiCache host tier | 13.5M KV tokens (82.9 GB) + mamba tier |
| GSM8K (1,319q, temp 0, zero-shot) | 97.2% regraded / 89.5% pinned grader |
| MTP acceptance | ~2.5 general content, ~6.0 math (257–267 tok/s C1) |
| Sustained C4 + capacity ladder | zero restarts, zero OOMs |

Full tables and method notes: [`BENCHMARKS.md`](BENCHMARKS.md).

## Quickstart

You need Linux x86_64, a CUDA 13-compatible driver, Docker with the NVIDIA
Container Toolkit, two visible SM120 GPUs, about 170 GB for the published
checkpoint. HiCache (the host-RAM prefix cache tier) is disabled by
default; enabling it needs free host RAM equal to the tier size
(`ENABLE_HICACHE=1`, `HICACHE_SIZE_GB=32` by default -- see the launcher).

### 1. Download the published checkpoint

```bash
export MODEL_REPO=ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
mkdir -p "$MODEL_DIR"
HF_XET_HIGH_PERFORMANCE=1 hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
```

### 2. Serve the qualified TP2 configuration

```bash
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v54
mkdir -p "$CACHE_DIR"
cat > adaptive.json <<'JSON'
{
  "1": {"candidate_steps": [3, 5], "up_hysteresis": 0.0, "down_hysteresis": -0.25, "ceiling_coeff": 0},
  "4": {"candidate_steps": [3, 5], "up_hysteresis": 0.0, "down_hysteresis": -0.25, "ceiling_coeff": 0}
}
JSON

docker run --rm --name glm53-flash --entrypoint sglang --gpus all \
  --shm-size 64g --ulimit memlock=-1 --publish 8000:8000 \
  --volume "$MODEL_DIR:/models/glm53:ro" \
  --volume "$CACHE_DIR:/root/.cache" \
  --volume "$PWD/adaptive.json:/etc/glm53-adaptive/adaptive.json:ro" \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8 \
  --env SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  --env SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
  --env SGLANG_PCIE_IPC_MAX_NUMEL=786432 \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  --env TILELANG_CACHE_DIR=/root/.cache/tilelang \
  --env TRITON_CACHE_DIR=/root/.cache/triton \
  ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.1.0 \
  serve /models/glm53 \
  --served-model-name glm-5.3-flash --host 0.0.0.0 --port 8000 \
  --tp 2 --quantization modelopt_mixed \
  --enable-multimodal --image-processor-backend torchvision --warmups serving_coverage \
  --mm-process-config '{"image": {"max_image_tokens": 3072}}' \
  --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion \
  --disable-custom-all-reduce \
  --attention-backend dsa --linear-attn-backend triton \
  --dsa-prefill-backend flashinfer_sparse_mla \
  --dsa-decode-backend flashinfer_sparse_mla \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --context-length 499712 --max-total-tokens 499712 \
  --mem-fraction-static 0.99 --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --max-running-requests 4 --max-mamba-cache-size 28 \
  # add: --enable-hierarchical-cache --hicache-size 32   (host prefix cache, optional) \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-bs-decode 1 2 3 4 \
  --speculative-algorithm EAGLE --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --speculative-adaptive-config /etc/glm53-adaptive/adaptive.json \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --enable-metrics --enable-cache-report
```

First boot compiles kernels into `$CACHE_DIR` (roughly 15–20 minutes); later
boots take about 8. Do not reuse a cache directory across image versions or
cache schemas. `examples/serve-glm53-flash.sh` wraps the same configuration.

Two configuration rules matter more than they look:

1. **`--cuda-graph-bs-decode` must enumerate every batch size up to
   `--max-running-requests`.** On this hybrid KDA model, decode batches that
   replay through a padded larger graph corrupt real requests' outputs
   (measured: GSM8K 68% vs 90% at temperature 0 with graphs captured only at
   bs 1 and 4). Exact graphs avoid the padding path entirely.
2. `--max-mamba-cache-size` is five recurrent-state slots per concurrent
   request (state plus MTP intermediates), so C4 needs 20.

## Reproducibility

`stack.lock.json` pins the official SGLang and FlashInfer base commits, the
checksummed integration patches in [`patches/`](patches/), the exact ModelOpt
commit and release tag, and the vendor base image digests.
`scripts/verify-patches.sh` re-fetches the official trees, applies the
patches, and asserts the exact resulting tree hashes. The carried patches
include the SM120 enablement and the CUDA-graph lifetime fixes submitted
upstream as
[flashinfer-ai/flashinfer#4827](https://github.com/flashinfer-ai/flashinfer/pull/4827),
[sgl-project/sglang#37168](https://github.com/sgl-project/sglang/pull/37168),
and
[sgl-project/sglang#37169](https://github.com/sgl-project/sglang/pull/37169),
plus three fixes to SGLang's `modelopt_mixed` loading path found while
qualifying this artifact (see `CHANGELOG.md`).

The full failure-and-fix narrative — the CUDA-graph workspace use-after-free
hunt, the padded-replay corruption, and the zero-acceptance MTP regression —
is preserved in [`CHANGELOG.md`](CHANGELOG.md) and the receipts under
[`evidence/`](evidence/).

Most users should download the published checkpoint as shown above. To
reproduce it from the pinned BF16 source, see
[`QUANTIZATION.md`](QUANTIZATION.md).

## Build from source

Verify before pushing:

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

Build locally:

```bash
podman build \
  --target runtime \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-glm53-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.1.1-rc.14 .
```

The release workflow refuses to overwrite an existing SemVer candidate tag.

## License

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Upstream SGLang,
FlashInfer, ModelOpt, and GLM-5.3-Flash retain their own licenses.
