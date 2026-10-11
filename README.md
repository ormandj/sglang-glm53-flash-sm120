# GLM-5.3-Flash on RTX PRO 6000 Blackwell GPUs

> [!IMPORTANT]
> **Upgrading from v0.5.x: replace your launch scripts and the TP4 checkpoint.** v0.6.0 is not a drop-in image update. The four-GPU profile now serves Z.ai's FP8 checkpoint `zai-org/GLM-5.3-Flash` instead of `nvidia/GLM-5.3-Flash-NVFP4`, with different memory and admission settings. Both launchers now use fixed native MTP, `examples/glm53-adaptive.json` is removed, and the kernel cache directories change to `-v90-tp2` and `-v90-tp4`. v0.5.x launch commands and the NVIDIA NVFP4 checkpoint were not qualified with this image. Download the checkpoint below and start from the v0.6.0 `examples/` scripts; see the [upgrade steps](CHANGELOG.md#upgrading-from-v05x).

A SGLang image for serving GLM-5.3-Flash on two or four NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs (SM120) over PCIe. Both profiles provide an OpenAI-compatible API, vision, reasoning, tool calling, FP8 KV cache and native speculative decoding.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.6.0` |
| TP4 checkpoint | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (FP8) |
| TP2 checkpoint | [`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) (4-bit experts) |

The current published stable image is `v0.6.0`. TP4 now serves Z.ai's FP8 checkpoint, which replaces the NVIDIA NVFP4 TP4 profile of `v0.5.0`. TP2 keeps the W4A16 checkpoint. The release refreshes SGLang, FlashInfer and Transformers, overlaps tensor-parallel all-reduces with compute during large TP4 prefills, splits long-context sparse-attention indexer work across GPUs, and runs both launchers with fixed three-step native MTP. See the [changelog](CHANGELOG.md) and [releases](https://github.com/ormandj/sglang-glm53-flash-sm120/releases).

| Default setting | TP2 | TP4 |
|---|---:|---:|
| GPUs | 2 × 96 GB SM120 | 4 × 96 GB SM120 |
| Checkpoint | W4A16 experts, FP8 projections | FP8 (Z.ai) |
| Per-request context limit | 524,288 tokens | 1,048,576 tokens |
| Shared device token pool | 524,288 tokens | 1,048,576 tokens |
| Maximum running requests | 4 | 16 |
| Prefill chunk / token budget | 4,096 | 8,192 |
| BF16 recurrent-state slots | 28 | 96 |
| Host cache | Optional, 32 GB/rank | Enabled, 40 GB/rank |
| Native MTP | Fixed: 3 steps, top-k 1, 4 verification tokens | Fixed: 3 steps, top-k 1, 4 verification tokens |

The token pool is shared across requests. Sixteen admitted requests do not each receive a 1M-token reservation; long prompts beyond the device pool are served from the host cache tier or wait for capacity.

## Requirements

- Linux x86_64, a CUDA 13 capable NVIDIA driver, Docker and the NVIDIA Container Toolkit.
- Two or four RTX PRO 6000 Blackwell 96 GB GPUs. Both profiles were validated on Max-Q cards at 250 W with PCIe Gen4 ×16 links, working peer access across CPU roots and no NVLink. Other GPU configurations are untested.
- Disk space for the chosen checkpoint and persistent compiled-kernel cache.
- Host RAM for model loading and serving, plus 160 GB for the default TP4 host cache. TP2's optional host cache adds 64 GB. These cache sizes are decimal GB; they are additional to process memory.

## Run with four GPUs

```bash
pip install -U huggingface_hub
export MODEL_DIR=/srv/models/GLM-5.3-Flash
HF_XET_HIGH_PERFORMANCE=1 hf download zai-org/GLM-5.3-Flash \
  --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir "$MODEL_DIR"
git clone https://github.com/ormandj/sglang-glm53-flash-sm120
cd sglang-glm53-flash-sm120
git checkout v0.6.0
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.6.0
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v90-tp4
./examples/serve-glm53-flash-tp4.sh
```

TP4 uses tensor parallel 4 and expert parallel 1 with the checkpoint's FP8 block-scaled weights and FP8 E4M3 KV cache.

## Run with two GPUs

```bash
pip install -U huggingface_hub
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
HF_XET_HIGH_PERFORMANCE=1 hf download ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO \
  --revision ee0989a944b0e213589191d7fca63af825a0741e --local-dir "$MODEL_DIR"
git clone https://github.com/ormandj/sglang-glm53-flash-sm120
cd sglang-glm53-flash-sm120
git checkout v0.6.0
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.6.0
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v90-tp2
./examples/serve-glm53-flash.sh
```

HiCache is off by default at TP2. Set `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` to enable 32 GB per rank. At TP4, set `ENABLE_HICACHE=0` to disable the default host tier. Use a fresh cache directory for this image and separate directories for the two profiles. The first startup compiles kernels; wait for `The server is fired up and ready to roll!`.

## Send a request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Explain KV cache paging in three sentences."}]}'
```

Both checkpoints always enable reasoning with their bundled chat template and return it in `reasoning_content`. Set `"reasoning_effort":"low"` or `"reasoning_effort":"high"` to select an effort level; the template otherwise defaults to `max`. Supply images as standard `image_url` content parts. The checkpoint's processor resizes images to its image-token budget; image tokens consume context capacity.

The launchers are plain `docker run` commands. [RUN.md](RUN.md) explains the profiles, native MTP settings and source builds.

## Measurements

Fixed-MTP measurements on RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 250 W. Decode uses 16K coding prompts and 4,096-token capped outputs including reasoning and a post-answer tail, which raises speculative acceptance; output rates are aggregate across the concurrency. Cold prefill is actual prompt tokens over time to first token.

| Profile | C1 output tok/s, mean / median | C1 target forwards/s | C4 output tok/s | C4 target forwards/s | Cold prefill, 32K / 128K prompt tok/s |
|---|---:|---:|---:|---:|---:|
| TP4, Z.ai FP8 | 222.0 / 212.6 | 77.8 | 428.1 | 37.5 | 10,616 / 10,134 |
| TP2, W4A16 | 192.0 / 181.8 | 64.9 | 391.1 | 32.8 | 6,160 / 6,086 |

TP4 cold prefill measured 10.1–10.6K prompt tokens per second from 8K to 128K prompts. At TP2, target forward rates rose 5.1% at C1 over the matched v0.5.0 measurement.

Measurements used v0.6.0 validation builds; the GHCR image is built separately from the same pinned inputs and was not separately benchmarked. These controlled measurements do not necessarily represent real-world performance. [BENCHMARKS.md](BENCHMARKS.md) provides all means and medians, C2 results, windows, sample counts, prefill shapes, GSM8K scores and configuration details.

## Limitations

- The supplied memory settings are tightly sized. Increasing pool size, concurrency or image budgets requires memory acceptance checks. Keep the prefill budget equal to the chunk size for the selected profile.
- High-resolution and multiple-image checks establish memory capacity; they do not establish OCR or photographic accuracy.
- Reasoning can consume the output budget before producing a final answer.
- Requesting input logprobs across a long prompt can exhaust GPU memory and restart the server. Score only the continuation at the prompt boundary.
- The TP2 checkpoint quantizes routed experts to 4-bit weights; its outputs differ from the FP8 and BF16 checkpoints.

## Reproducibility

[stack.lock.json](stack.lock.json) pins the official upstream commits, checksummed integration patches, ModelOpt, DeepGEMM, Transformers and vendor base-image digests. [scripts/verify-patches.sh](scripts/verify-patches.sh) fetches the official source trees, applies the patches and verifies the resulting tree hashes. The vendor base supplies the CUDA/PyTorch dependency stack; it does not establish SGLang source provenance.

[QUANTIZATION.md](QUANTIZATION.md) describes how the TP2 W4A16 checkpoint was produced. TP4 uses Z.ai's published FP8 checkpoint directly.

## License

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Upstream SGLang, FlashInfer, ModelOpt and GLM-5.3-Flash retain their own licenses.
