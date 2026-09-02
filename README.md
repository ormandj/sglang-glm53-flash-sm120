# SGLang for GLM-5.3-Flash on SM120

A reproducible Linux x86_64 image for serving the published
[`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO)
checkpoint on two NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) GPUs with
SGLang, TP2, no NVLink required.

**Hugging Face model:**
[`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO)

Current release: `v0.1.1` (stable; digest-identical promotion of
`v0.1.1-rc.16`) for the W4A16 NVFP4 K32 experts + FP8 weight-only
checkpoint (internal build name `sglang-glm53-flash-sm120:v0.1.1-rc.16`):

```text
ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.1.1
```

`v0.1.1` fixes the no-restore Xid 31 fault in the DSA top-k kernel
(reported by @bold84 and @sousekd, fixed by @bold84), raises the mamba
pool so four simultaneous cold requests admit, serves 3840x2160 and
7680x4320 images at the model's 8,000-token budget without OOM, warms
every serving shape before ready (first sampled chat 1.4 s instead of
75 s on a new image), keeps image preprocessing on the CPU, and moves to
upstream SGLang and FlashInfer main of 2026-09-02 with every downstream
change submitted upstream (`sgl-project/sglang#37534` to `#37541`).
Validation on rc.16 (the promoted candidate): 9-minute first boot with
zero late kernel loads at ready, 4K and 8K images 3.4 s and 3.0 s, four
concurrent cold 4k-token prefills 3.3 s each, GSM8K 98/100 on a
100-question subset. `v0.1.0` remains available.

## What you get

The numbers below were measured on `v0.1.1` (2026-09-02) unless a row says
otherwise.

- **C4 serving with a 499,712-token KV pool and context limit**: four
  concurrent agentic requests sharing a full ~500k-token budget, with a
  494,592-token single-request prefill served at 99.2% pool usage.
- **HiCache host tier**: 13.5M KV tokens (82.9 GB host memory) plus a mamba
  state tier, so evicted long prefixes resume from RAM instead of
  recomputing.
- **Adaptive EAGLE/NextN MTP speculative decoding** (candidate steps [3,5],
  acceptance-driven) plus PCIe IPC allreduce for TP2 small reduces:
  decode plateaus of 174.4 tok/s at C1 (52.7 forwards/s), 252.1 at C2
  (38.4 fwd/s) and 298.8 at C3 (33.0 fwd/s); cold 200k prefill 6,000 tok/s.
  The math-content C1 figure of 257 to 267 tok/s is a v0.1.0 measurement.
- **Vision intact**, exercised through the OpenAI-compatible image input.
- **GSM8K 96.9%** (1,278/1,319, zero-shot, temperature 0, position-based
  answer extraction; 89.0% under the raw last-number grader), within
  repetition spread of v0.1.0's 97.2% and the BF16-attention control's
  97.0%.
- **Reproducible end to end**: exact upstream SGLang/FlashInfer commits plus
  checksummed patches (`stack.lock.json`, `scripts/verify-patches.sh`), the
  quantization producers (`quantization/`), the benchmark harness
  (`bench/`), and raw receipts (`evidence/`).

### Performance and capacity at a glance

All rows were measured on `v0.1.1` (2026-09-02) in the shipped
configuration with the vendored harness (pinned aiperf 0.12.0): decode is
the n=5 coding corpus at 4,096 output tokens, temperature 0, cohort-only
cells, warm server; every repetition is analyzer-valid. Tokens per second
is after MTP acceptance; forwards per second is the engine step rate.

| Decode (n=5, OLS plateau over server counters) | tok/s mean (median) | Forwards/s mean | Accepted tok/fwd/req |
|---|---:|---:|---:|
| C1 | 174.4 (175.1) | 52.7 | 3.5 |
| C2 | 252.1 (244.5) | 38.4 | 3.2 |
| C3 | 298.8 (297.7) | 33.0 | 3.0 |
| C4 | no qualified cell: at most 3 requests decode at once (mamba admission, see BENCHMARKS.md); 8/8 four-request waves at 255 to 269 aggregate tok/s |

| Cold prefill (5 requests each, C1, prompt tokens / TTFT) | Rate |
|---|---:|
| 8k | 5,263 tok/s |
| 32k | 5,903 tok/s |
| 64k | 5,918 tok/s |
| 128k | 5,918 tok/s |
| 200k (ladder, single request) | 6,000 tok/s |

| Capacity and quality | Result |
|---|---|
| KV pool / context limit | 499,712 tokens, four running requests |
| Largest single prefill served | 494,592 tokens (pool usage 99.2%, on a 200k cached prefix) |
| HiCache host tier (32 GB setting) | 2.65M KV tokens (20.9 GB) + 11.2 GB mamba tier |
| GSM8K (1,319q, temp 0, zero-shot) | 96.9% regraded / 89.0% pinned grader |
| MTP acceptance (coding corpus) | 2.7 to 4.2 accepted tokens per forward per request |
| Large images | 3840x2160 in 3.4 s, 7680x4320 in 3.0 s at the model's 8,000-token budget |
| Ladder, waves, gates, GSM8K | zero restarts, zero OOM errors; four recoverable allocator segment-mapping warnings at 99% pool usage |

Full tables and method notes: [`BENCHMARKS.md`](BENCHMARKS.md).

## Carried upstream changes

The image is official SGLang and FlashInfer `main` plus checksummed patches
(`patches/`, pinned in `stack.lock.json`). Everything in those patches that
is not SM120-specific glue is an open upstream pull request; the image
carries the exact PR commits. When one merges, it leaves the patch at the
next base refresh.

SGLang (base `main` 1109e44305, 2026-09-02):

| PR | What it carries |
|---|---|
| [sgl-project/sglang#36507](https://github.com/sgl-project/sglang/pull/36507) | GLM-5.3-Flash model support (the series branch, at head 515e865189) |
| [sgl-project/sglang#36904](https://github.com/sgl-project/sglang/pull/36904) | TileLang fp8_e4m3 KV cache on CUDA (raw layout) |
| [sgl-project/sglang#36661](https://github.com/sgl-project/sglang/pull/36661) | overlap batch snapshot lifetime tied to result completion |
| [sgl-project/sglang#36696](https://github.com/sgl-project/sglang/pull/36696) | mamba radix cache split-node key fix |
| [sgl-project/sglang#36821](https://github.com/sgl-project/sglang/pull/36821) | ReplaySSM ring-write in the fused KDA chain-verify kernel |
| [sgl-project/sglang#37168](https://github.com/sgl-project/sglang/pull/37168) | full CUDA-graph capture owners (MHC and DSA tensors replayed by captured graphs) |
| [sgl-project/sglang#37169](https://github.com/sgl-project/sglang/pull/37169) | opt-in allocator-history forensics snapshots |
| [sgl-project/sglang#37534](https://github.com/sgl-project/sglang/pull/37534) | HiCache host pool rows sized to packed DSA KV rows |
| [sgl-project/sglang#37535](https://github.com/sgl-project/sglang/pull/37535) | opt-in token-blocked KDA extend (`SGLANG_KDA_EXTEND_BLOCK_TOKENS`) |
| [sgl-project/sglang#37536](https://github.com/sgl-project/sglang/pull/37536) | raw multimodal features released from the device after encoding |
| [sgl-project/sglang#37537](https://github.com/sgl-project/sglang/pull/37537) | `--mm-preprocessing-device` for the base visual preprocessing path |
| [sgl-project/sglang#37538](https://github.com/sgl-project/sglang/pull/37538) | env-gated extend memory profiler |
| [sgl-project/sglang#37539](https://github.com/sgl-project/sglang/pull/37539) | GLM-5-Next vision-tower attention and compiled-activation precompile at startup |
| [sgl-project/sglang#37540](https://github.com/sgl-project/sglang/pull/37540) | kpool top-k transform: clipped stage-1 bins and bounded selection stores (by @bold84) |
| [sgl-project/sglang#37541](https://github.com/sgl-project/sglang/pull/37541) | opt-in `serving_coverage` request warmup |

FlashInfer (base `main` c5ff6f48, 2026-09-02):

| PR | What it carries |
|---|---|
| [flashinfer-ai/flashinfer#4802](https://github.com/flashinfer-ai/flashinfer/pull/4802) | native SM120 sparse-MLA runner with GLM NoPE rows (at head 98bcd8501b) |
| [flashinfer-ai/flashinfer#4687](https://github.com/flashinfer-ai/flashinfer/pull/4687) | W4A16 large weight bank addressing |
| [flashinfer-ai/flashinfer#4827](https://github.com/flashinfer-ai/flashinfer/pull/4827) | SM12x MoE workspaces kept alive while captured CUDA graphs reference them |

Downstream-only in the patches (no upstream PR): the SM120 W4A16 MoE and
NoPE-row integration, the ModelOpt E4M3-K32 W4A16 weight preparation, the
adaptive-MTP chain-buffer pinning, and the Triton late-load diagnostic
armed after the serving warmup. Merged upstream and already in the base:
[sgl-project/sglang#37317](https://github.com/sgl-project/sglang/pull/37317),
[#36958](https://github.com/sgl-project/sglang/pull/36958),
[#36798](https://github.com/sgl-project/sglang/pull/36798).

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
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v55
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
patches, and asserts the exact resulting tree hashes. The patches are the
open upstream pull requests listed under "Carried upstream changes" plus
the SM120 glue, and three fixes to SGLang's `modelopt_mixed` loading path
found while qualifying this artifact (see `CHANGELOG.md`).

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
  -t sglang-glm53-flash-sm120:v0.1.1-rc.16 .
```

The release workflow refuses to overwrite an existing SemVer candidate tag.

## License

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Upstream SGLang,
FlashInfer, ModelOpt, and GLM-5.3-Flash retain their own licenses.
