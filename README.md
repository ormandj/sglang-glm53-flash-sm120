# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs

A ready-to-run SGLang image and a matching quantized checkpoint for serving GLM-5.3-Flash on two NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) GPUs over PCIe. Download the checkpoint and run the launcher for an OpenAI-compatible server with a 450,560-token context, four concurrent requests, speculative decoding, vision input, reasoning and tool calling.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.3.2` |
| Checkpoint | [`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) on Hugging Face |
| Hardware | 2x RTX PRO 6000 Blackwell (SM120), tensor parallel 2, PCIe |

The current published stable image is `v0.3.2`. This version updates SGLang to the merged GLM-5.3-Flash support while retaining the model, quantization and serving settings. See the [changelog](CHANGELOG.md) and [published releases](https://github.com/ormandj/sglang-glm53-flash-sm120/releases) for release details.

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
   export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.3.2
   export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v69
   ./examples/serve-glm53-flash.sh
   ```

   To enable 32 GB of host cache per rank, set
   `ENABLE_HICACHE=1 HICACHE_SIZE_GB=32` when running the launcher.
   The first boot compiles kernels into `CACHE_DIR` and takes about 10 to 20
   minutes; later boots take about 8. Use a fresh `CACHE_DIR` for every
   image version. The server is ready when the log prints
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
to see or change every flag. [`RUN.md`](RUN.md) has a reduced first-boot
profile. Two settings matter more than they look:

- `--cuda-graph-bs-decode` lists every batch size up to `--max-running-requests`, so each decode batch uses a graph captured for its size.
- `--max-mamba-cache-size` is recurrent-state slots, not KV cache. Each live
  request uses four to five, so the launcher ships 28 for four requests.

## What to expect

The following measurements use two RTX PRO 6000 Blackwell Max-Q GPUs at 300 W, with the launcher settings and 32 GB of HiCache per rank. They were collected on the `v0.3.2` validation build; the GHCR image was built separately from the same pinned inputs. [Benchmark provenance and quality results](BENCHMARKS.md#v032-qualification-2026-09-06) identify the tested artifact and limitations.

| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 215.5 | 202.6 | 60.77 | 61.38 | 3.60 / 3.42 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 290.2 | 293.8 | 49.11 | 49.16 | 3.02 / 3.08 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 360.3 | 356.6 | 39.20 | 38.98 | 3.11 / 3.12 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 406.9 | 410.2 | 33.31 | 33.62 | 3.06 / 3.03 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,193.5 | 5,211.0 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 5,877.4 | 5,860.4 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 5,921.4 | 5,907.2 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 5,923.1 | 5,903.7 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 10.7-29.7 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.

Decode tok/s is aggregate output after MTP, including reasoning, across the stated number of concurrent requests. Forward passes/s counts target-model iterations. Decode uses a fixed 4,096-token response window; its post-answer tail can increase speculative acceptance. Prefill tok/s is the per-request prompt-token count divided by time to first token, reported as mean and median over five cold requests. These controlled measurements do not necessarily represent real-world performance. [Latency and supplemental decode measurements](BENCHMARKS.md#v032-qualification-2026-09-06) provide the other measured views.

## Limitations

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
  -t sglang-glm53-flash-sm120:v0.3.2-rc.1 .
```

The vendor base image supplies the pinned CUDA/PyTorch dependency stack; the SGLang and FlashInfer source trees are verified separately as described above.

## License

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Upstream SGLang,
FlashInfer, ModelOpt and GLM-5.3-Flash retain their own licenses.

## Building from source

This checkout builds `sglang-glm53-flash-sm120:v0.4.0-rc.5`. See [RUN.md](RUN.md) for the matching launcher and cache directory. Published-image measurements above apply to `v0.3.2`.
