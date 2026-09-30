# GLM-5.3-Flash on RTX PRO 6000 Blackwell GPUs

A SGLang image for serving GLM-5.3-Flash on two or four NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs (SM120) over PCIe. Both profiles provide an OpenAI-compatible API, vision, reasoning, tool calling, FP8 KV cache and native speculative decoding.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0` |
| TP2 checkpoint | [`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) |
| TP4 checkpoint | [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) |

The current published stable image is `v0.5.0`. It refreshes SGLang, FlashInfer, ModelOpt, DeepGEMM and Transformers, adds the NVIDIA NVFP4 TP4 profile, and corrects recurrent-state checkpoint handling during blocked prefills. See the [changelog](CHANGELOG.md) and [releases](https://github.com/ormandj/sglang-glm53-flash-sm120/releases).

**TP2:** no decode or prefill improvement over v0.4.3 has been established. The current measured rates are lower than the previous published rates, with different power and cache settings: 250 W without HiCache now, versus 300 W with HiCache before.

**TP4:** the same cold 131,173-token prompt at 8,192-token prefills reached its first token in 18.161 seconds on v0.4.3 and 15.549 seconds with v0.5.0 and corrected PCIe peer transport, equivalent to **16.8% higher prompt-processing throughput**. This is a single diagnostic request per configuration with different serving capacities. A decode speedup over v0.4.3 has not been established. See the [measured rates](#measurements).

| Default setting | TP2 | TP4 |
|---|---:|---:|
| GPUs | 2 × 96 GB SM120 | 4 × 96 GB SM120 |
| Per-request context limit | 524,288 tokens | 1,048,576 tokens |
| Shared device token pool | 524,288 tokens | 2,621,440 tokens |
| Maximum running requests | 4 | 32 |
| Prefill chunk / token budget | 4,096 | 8,192 |
| BF16 recurrent-state slots | 28 | 224 |
| Host cache | Optional, 32 GB/rank | Enabled, 40 GB/rank |

The token pool is shared across requests. A 32-request admission limit does not reserve 32 full 1M-token contexts. TP2 keeps the previous W4A16 checkpoint; [fixed-MTP measurements](BENCHMARKS.md#v050-tp2-w4a16-fixed-native-mtp) cover C1, C2 and C4 decode and cold prefill up to 128K.

## Requirements

- Linux x86_64, a CUDA 13 capable NVIDIA driver, Docker and the NVIDIA Container Toolkit.
- Two or four RTX PRO 6000 Blackwell 96 GB GPUs. The TP4 profile was validated on Max-Q cards at 250 W with PCIe Gen4 ×16 links, working peer access across CPU roots and no NVLink. Other GPU configurations are untested.
- Disk space for the chosen checkpoint and persistent compiled-kernel cache.
- Host RAM for model loading and serving, plus 160 GB for the default TP4 host cache. TP2's optional host cache adds 64 GB. These cache sizes are decimal GB; they are additional to process memory.

## Run with four GPUs

```bash
pip install -U huggingface_hub
export MODEL_DIR=/srv/models/GLM-5.3-Flash-NVFP4
HF_XET_HIGH_PERFORMANCE=1 hf download nvidia/GLM-5.3-Flash-NVFP4 \
  --revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3 --local-dir "$MODEL_DIR"
git clone https://github.com/ormandj/sglang-glm53-flash-sm120
cd sglang-glm53-flash-sm120
git checkout v0.5.0
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v88-tp4
./examples/serve-glm53-flash-tp4.sh
```

The launcher includes the metadata override needed to load NVIDIA's BF16 native MTP layer without quantizing it. Keep that override and `--speculative-draft-model-quantization unquant` together. TP4 uses tensor parallel 4 and expert parallel 1.

## Run with two GPUs

Use the previous W4A16 checkpoint with the TP2 launcher:

```bash
pip install -U huggingface_hub
export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
HF_XET_HIGH_PERFORMANCE=1 hf download ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO --local-dir "$MODEL_DIR"
git clone https://github.com/ormandj/sglang-glm53-flash-sm120
cd sglang-glm53-flash-sm120
git checkout v0.5.0
export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v88-tp2
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

**TP4, NVIDIA NVFP4:** 197.24 mean / 197.40 median output tok/s after MTP and 75.73 mean / 76.51 median target forwards/s at C1 on four RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 250 W.

**TP2, previous W4A16 checkpoint:** target forwards/s are lower than the [previous published v0.4.3 results](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.3/BENCHMARKS.md):

| Concurrency | v0.4.3 mean forwards/s | v0.5.0 mean forwards/s | Observed change |
|---|---:|---:|---:|
| C1 | 66.91 | 61.78 | −7.7% |
| C2 | 51.07 | 47.23 | −7.5% |
| C4 | 35.34 | 32.29 | −8.6% |

Both use fixed three-step MTP. The previous tests used 300 W and 32 GB/rank HiCache; the current tests use 250 W without HiCache. These rates do not isolate a software regression or speedup.

Current TP2 mean aggregate output is 181.60 / 287.55 / 381.50 tok/s at C1 / C2 / C4. Cold prefill averaged 5,190.98 prompt tok/s at 8K and 6,076.75–6,132.49 prompt tok/s at 32K–128K.

These results use fixed three-step MTP with adaptive switching disabled, 16K coding prompts and 4,096-token capped outputs including reasoning and a post-answer tail. Measurements used a v0.5.0 validation build; the GHCR image is built separately from the same pinned inputs and was not separately benchmarked. [BENCHMARKS.md](BENCHMARKS.md) provides all means/medians, target-forward rates, windows, sample counts and configuration details.

## Limitations

- The supplied memory settings are tightly sized. Increasing pool size, concurrency or image budgets requires memory acceptance checks. Keep the prefill budget equal to the chunk size for the selected profile.
- High-resolution and multiple-image checks establish memory capacity on the TP4 profile; they do not establish OCR or photographic accuracy.
- Reasoning can consume the output budget before producing a final answer.
- Requesting input logprobs across a long prompt can exhaust GPU memory and restart the server. Score only the continuation at the prompt boundary.
- HiCache does not support FP4 MLA KV storage with separate scale buffers or multiple mismatched draft runners. This concerns KV storage, not the supplied FP4 weight checkpoints.

## Reproducibility

[stack.lock.json](stack.lock.json) pins the official upstream commits, checksummed integration patches, ModelOpt, DeepGEMM, Transformers and vendor base-image digests. [scripts/verify-patches.sh](scripts/verify-patches.sh) fetches the official source trees, applies the patches and verifies the resulting tree hashes. The vendor base supplies the CUDA/PyTorch dependency stack; it does not establish SGLang source provenance.

[QUANTIZATION.md](QUANTIZATION.md) describes the previous owner W4A16 checkpoint used at TP2. TP4 uses NVIDIA's published NVFP4 checkpoint directly.

## License

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Upstream SGLang, FlashInfer, ModelOpt and GLM-5.3-Flash retain their own licenses.
