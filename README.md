# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs

A ready-to-run SGLang image and a matching quantized checkpoint for serving GLM-5.3-Flash on two NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) GPUs over PCIe. Download the checkpoint and run the launcher for an OpenAI-compatible server with a 524,288-token context limit and shared device token pool, up to four concurrent requests within that pool, speculative decoding, vision input, reasoning and tool calling.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.4.2` |
| Checkpoint | [`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) on Hugging Face |
| Hardware | 2x RTX PRO 6000 Blackwell (SM120), tensor parallel 2, PCIe |

The current published stable image is `v0.4.2`. It corrects HiCache recurrent-state checkpoint boundaries and shared-buffer ownership, requires the corrected native FlashInfer NoPE execution contract, and refreshes SGLang, FlashInfer and the carried fixes. The 524,288-token shared device pool, four-request admission, compact FP8 cache, reasoning, vision and native speculative decoding remain supported. See the [changelog](CHANGELOG.md) and [published releases](https://github.com/ormandj/sglang-glm53-flash-sm120/releases) for details.

The older `v0.2.1` image has a long-prefix HiCache corruption defect. Keep HiCache disabled if continuing to use that version.

## Requirements

- Linux x86_64 with a CUDA 13 capable driver, Docker and the NVIDIA
  Container Toolkit.
- Two visible SM120 GPUs. Other GPU pairs are untested.
- About 170 GB of disk for the checkpoint and a few GB for the kernel cache.
- Optional: 64 GB for the TP2 HiCache pools at 32 GB per rank, plus host RAM
  for the model-loading and serving processes. HiCache is off by default in
  the launcher; the benchmark configuration enables it at 32 GB per rank.

## Run it

1. Download the checkpoint.

   ```bash
   pip install -U huggingface_hub
   export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
   HF_XET_HIGH_PERFORMANCE=1 hf download ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO --local-dir "$MODEL_DIR"
   ```

2. Start the server with the launcher from this repository. Enable the optional
   HiCache settings below to match the benchmark configuration.

   ```bash
   git clone https://github.com/ormandj/sglang-glm53-flash-sm120
   cd sglang-glm53-flash-sm120
   export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.4.2
   export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v81
   ./examples/serve-glm53-flash.sh
   ```

   To enable 32 GB of host cache per rank, set
   `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` when running the launcher.
   The first boot compiles kernels into `CACHE_DIR`. Use a fresh `CACHE_DIR` for this image version and wait for warmup to finish. The server is ready when the log prints
   `The server is fired up and ready to roll!`.

3. Send a request. The API is OpenAI-compatible on port 8000 and the model
   name is `glm-5.3-flash`.

   ```bash
   curl -s http://localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Explain KV cache paging in three sentences."}]}'
   ```

   Reasoning is on by default and returned in `reasoning_content`. To turn
   it off for a request, add `"chat_template_kwargs": {"enable_thinking": false}`.
   Images go in as standard `image_url` content parts, up to 8,000 image
   tokens per request.

The launcher is a plain `docker run`; read [`examples/serve-glm53-flash.sh`](examples/serve-glm53-flash.sh)
to see or change every flag. [`RUN.md`](RUN.md) describes the serving configuration. Keep these settings together:

- Prefill uses breakable CUDA graphs for single-request 64- and 128-token tails; longer prefills retain 4,096-token chunks.
- `--cuda-graph-bs-decode` lists every batch size up to `--max-running-requests`, so each decode batch uses a graph captured for its size.
- `--max-mamba-cache-size` is recurrent-state slots, not KV cache. Each live
  request uses four to five, so the launcher ships 28 for four requests.

To build this source locally as `sglang-glm53-flash-sm120:v0.4.3-rc.2`, follow [RUN.md](RUN.md).

## What to expect

Measurements used a v0.4.2 validation build on two RTX PRO 6000 Blackwell Max-Q 96 GB GPUs at 300 W, tensor parallel 2 over PCIe. The GHCR image is built separately from the same pinned inputs and was not separately benchmarked. The configuration used W4A16 experts, FP8 KV, a 524,288-token shared device pool, four running requests, 4,096-token prefill chunks, 28 recurrent-state slots and 32 GB of HiCache per rank. Performance comparisons use fixed native MTP with three draft steps, top-k one and four verification tokens, with adaptive switching disabled. The launcher uses adaptive MTP for serving. The source inputs are pinned in [this release's stack lock](https://github.com/ormandj/sglang-glm53-flash-sm120/blob/v0.4.2/stack.lock.json).

| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 196.1 | 193.0 | 66.78 | 66.72 | 2.95 / 2.96 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 306.0 | 306.8 | 50.99 | 50.89 | 3.00 / 3.02 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 369.0 | 370.2 | 41.08 | 41.03 | 2.99 / 3.05 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 411.0 | 408.4 | 35.38 | 35.16 | 2.90 / 2.91 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,647.7 | 5,659.0 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 6,352.0 | 6,359.6 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 6,361.1 | 6,345.1 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 6,346.7 | 6,319.5 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 13.3-29.3 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.

Decode tok/s is aggregate output after MTP, including reasoning, across the stated number of concurrent requests. Forward passes/s counts target-model iterations. Every decode response reaches a deliberate 4,096-token output cap with `ignore_eos`; the post-answer tail can increase speculative acceptance, so this is not completed-answer throughput. Prefill tok/s is each cold request's prompt-token count divided by time to first token, summarized over five requests per length. These controlled measurements do not necessarily represent real-world performance.

Compared with v0.4.1, mean output rates changed by +8.08% at C1, -1.95% at C2, -0.69% at C3 and +0.15% at C4. Mean target forward rates were 0.23%-0.79% lower; mean and median cold-prefill rates differed by at most 0.56%. Output-rate changes also reflect speculative acceptance in the fixed window, including its post-answer tail. Five sequential repetitions within one startup per image do not establish an isolated speedup or statistical significance. [BENCHMARKS.md](BENCHMARKS.md) contains the full comparison, methodology and adaptive-serving quality results.

## Limitations

- Reasoning can exhaust the output budget without producing a final answer. This occurred in two of 1,319 GSM8K requests at a 16,384-token budget; see [BENCHMARKS.md](BENCHMARKS.md) for completion and quality results.
- Custom HiCache configurations with one differently stored MLA draft use additional host memory for its sidecars. Mismatched multiple draft runners and FP4 MLA KV storage with separate scale buffers are unsupported; this does not restrict the supplied W4A16 weight quantization.
- A request for input logprobs spanning a long prompt can still OOM the scheduler and restart the container. Score only the continuation at the prompt boundary.
- Memory is tightly sized at `mem-fraction-static=0.99`. Prefill, vision and runtime compilation share the remaining headroom; keep `max-prefill-tokens` equal to the 4,096-token chunk size. Increasing the pool, concurrency or image budget requires new memory acceptance tests.
- Images use approximately one token per 28x28 pixels up to the checkpoint's 8,000-token limit. Larger images are resized by the processor; image tokens consume context capacity.
- Small bookkeeping kernels can log `device-loaded after serving started` for alignment-specialized variants loaded from the persistent cache in milliseconds. Such lines remain present on this release. A first request delayed by seconds of compilation needs investigation.
- Measurements cover this two-GPU SM120 profile. Other cards and tensor-parallel sizes are untested.

## Reproducibility and building

`stack.lock.json` pins the SGLang and FlashInfer base commits, the
checksummed patches in [`patches/`](patches/), the ModelOpt commit and the
vendor base image digests. `scripts/verify-patches.sh` re-fetches the
official trees, applies the patches and asserts the resulting tree hashes.
[`QUANTIZATION.md`](QUANTIZATION.md) reproduces the checkpoint from the BF16
source with the producers in [`quantization/`](quantization/).

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
podman build --target runtime \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-glm53-flash-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.4.2 .
```

The vendor base image supplies the pinned CUDA/PyTorch dependency stack; the SGLang and FlashInfer source trees are verified separately as described above.

## License

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Upstream SGLang,
FlashInfer, ModelOpt and GLM-5.3-Flash retain their own licenses.
