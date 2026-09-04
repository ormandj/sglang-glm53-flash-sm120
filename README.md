# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs

A ready-to-run SGLang image and a matching quantized checkpoint for serving
GLM-5.3-Flash on two NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) GPUs over
PCIe. No NVLink, no source build, no patching: download the checkpoint, run
one command, and you have an OpenAI-compatible server with a 450,560-token
context, four concurrent requests, speculative decoding, vision input,
reasoning and tool calling.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.2.0` |
| Checkpoint | [`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) on Hugging Face |
| Hardware | 2x RTX PRO 6000 Blackwell (SM120), tensor parallel 2, PCIe |
| Quality | GSM8K 96.9% (1,278 of 1,319, zero-shot, greedy) |

The published, qualified image remains `v0.2.0`. This repository currently
builds `v0.2.1-rc.6`, an unbuilt source candidate rebased onto the official
SGLang and FlashInfer `main` heads fetched on 2026-09-03. It has no performance,
quality, stability, or capacity claims. Its immediate predecessor,
`v0.2.1-rc.5`, was built and qualified internally on quasar at
`sha256:289e83c983fb951ed5265de80cca3dc0412cc0cd43cf2190296a7f8190c38f69`
but was not promoted or published.

The checkpoint keeps the routed experts in W4A16 NVFP4 (K32 blocks) and the
attention, shared experts and MTP draft layer in FP8 or BF16. Upstream SGLang
cannot serve GLM-5.3-Flash on SM120 yet; this image is upstream `main` plus
the open pull requests listed under [Carried upstream changes](#carried-upstream-changes).

## Requirements

- Linux x86_64 with a CUDA 13 capable driver, Docker and the NVIDIA
  Container Toolkit.
- Two visible SM120 GPUs. Other GPU pairs are untested.
- About 170 GB of disk for the checkpoint and a few GB for the kernel cache.
- Optional: 32 GB of free host RAM for the HiCache prefix tier (off by
  default in the launcher, on in the measured configuration).

## Run it

1. Download the checkpoint.

   ```bash
   pip install -U huggingface_hub
   export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
   HF_XET_HIGH_PERFORMANCE=1 hf download ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO --local-dir "$MODEL_DIR"
   ```

2. Start the server with the launcher from this repository. It runs the exact
   configuration the numbers below were measured with.

   ```bash
   git clone https://github.com/ormandj/sglang-glm53-flash-sm120
   cd sglang-glm53-flash-sm120
   export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.2.0
   export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v56
   ENABLE_HICACHE=1 ./examples/serve-glm53-flash.sh
   ```

   Drop `ENABLE_HICACHE=1` if the host does not have 32 GB of RAM to spare.
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

- `--cuda-graph-bs-decode` lists every batch size up to
  `--max-running-requests`, so no decode batch replays through a padded
  graph. On builds before the 2026-09-02 upstream base, padded replay
  corrupted outputs on this hybrid model (GSM8K 68% against 90% with graphs
  at batch 1 and 4 only); on v0.1.4 the same test scores 98.3% on 300
  questions, so the exact list is kept as a precaution, not a requirement.
- `--max-mamba-cache-size` is recurrent-state slots, not KV cache. Each live
  request uses four to five, so the launcher ships 28 for four requests.

## What to expect

Measured on 2026-09-03 on `v0.2.0-rc.1` with the launcher configuration plus
HiCache, using the vendored harness in [`bench/`](bench/) on a warm server,
at the 450,560-token pool. Full tables, per-repetition numbers and method
notes are in [`BENCHMARKS.md`](BENCHMARKS.md); the raw receipts and gate
summaries for v0.2.0 are in [`evidence/`](evidence/) (copied from the primary
repository, which holds every experiment behind them).

**Decode.** Coding prompts of about 19k tokens, 4,096 output tokens, greedy,
five repetitions per cell. Tokens per second counts accepted speculative
tokens; forwards per second is the engine step rate; the last column is how
many tokens each request accepted per engine forward. Rates are fixed-window
measurements on synthetic prompts, not a promise of application throughput.

| Concurrent requests | Output tok/s, mean (median) | Forwards/s, mean (median) | Accepted tok/forward/request | v0.1.4 on the same gate (tok/s at fwd/s, accepted) |
|---|---:|---:|---:|---:|
| 1 | 264.1 (283.1) | 54.6 (52.3) | 4.9 | 156.6 at 54.5, 2.9 |
| 2 | 306.9 (306.9) | 39.1 (38.1) | 3.9 | 237.1 at 39.2, 3.0 |
| 3 | 415.7 (387.1) | 31.1 (30.9) | 4.5 | 319.0 at 30.6, 3.5 |
| 4 | 386.4 (385.8) | 31.3 (31.5) | 3.1 | 371.9 at 28.8, 3.3 |

At four requests the engine step rate is up 9%. At one to three requests the
step rate is the same on both images and the difference is in the accepted
column (a mean): the harness decodes a fixed 4,096-token window with
`ignore_eos`, and once the greedy answer ends the drafter predicts the tail
at 5.8-6.0 of 6, so that column depends on where each answer ended. The
like-for-like engine comparison is the step rate at equal acceptance, which
was measured at concurrency 1 only (the probe below); no equal-acceptance
estimate exists for two or three requests.

On a fixed-acceptance workload (a repetitive ledger prompt, 1k or 19k of
context, 2,048 output tokens, six runs after a 150 s thermal soak), the
engine step rate at concurrency 1 is 67 forwards/s and 154 tok/s at 2.3
accepted tokens per forward; v0.1.4 measured 56 forwards/s and 130 tok/s at
the same acceptance on the same probe and protocol (+19%; run-to-run noise
across boots is about 3%).

**Prefill.** Five cold, cache-busted requests per length, one at a time
(within 0.6% of the v0.1.1 panel).

| Prompt | Prompt tok/s | Time to first token |
|---|---:|---:|
| 8k | 5,245 | 1.6 s |
| 32k | 5,871 | 5.6 s |
| 64k | 5,890 | 11.1 s |
| 128k | 5,893 | 22.2 s |
| 200k | 6,000 (v0.1.1) | 34.1 s |

**Capacity and quality.**

| | |
|---|---|
| Context and KV pool | 450,560 tokens shared by up to four requests |
| Largest prompt served | 436,295 tokens plus a full-budget image at 99% pool usage (v0.1.3) |
| Concurrent long prompts | 4 x 111,616 tokens cold at once, cache flushed first (19.3 s to first token); 3 x 146,460 plus a full-budget 3840x2160 image |
| HiCache host tier (32 GB) | 2.55M KV tokens (packed MTP layers) plus the recurrent-state tier |
| GSM8K | 296/300 on the v0.2.0-rc.1 300-question subset, greedy, four concurrent (v0.1.4 on the same run: 296/300); 96.9% on 1,319 questions on v0.1.1 |
| Images | 3840x2160 in 2.3-3.5 s, alone and with three 128k prompts resident; ten in a row |
| Stability | zero restarts and zero OOM errors across the decode, prefill, capacity, vision and GSM8K runs; a request for input logprobs over a long prompt can still OOM the scheduler and restart the container (shared with v0.1.4, see `BENCHMARKS.md`) |

**Limits worth knowing.**

- Memory is sized to the edge on purpose. Weights take 84.7 GB per GPU and
  the 450,560-token KV pool only 4.0 GB, so the pool is not the lever
  people expect: every 100k tokens of pool is 0.88 GB. v0.1.3 moved the
  pool from 499,712 to 450,560 tokens after two crashes were found at the
  larger size (a full-budget image encode while three long requests were
  decoding, and four cold 95k-token prompts at once); at 450,560 every
  stress shape in `BENCHMARKS.md` passes with about 0.3 GiB to spare. If
  you need more context, lower the image budget instead: a full-budget
  image costs 0.85 GiB of transient encode memory, the same as about 97k
  pool tokens.
- Images cost 28x28 pixels per token up to `max_image_tokens` = 8,000 in
  the checkpoint's processor config (6.3 megapixels, so a 3840x2160 frame
  is scaled to about 3339x1878). A 1080p screenshot is about 2,650 tokens.
  For agents, send screenshots at 1080p to 1440p and crop-zoom for detail;
  only sources above 6 megapixels pay the full 8,000 tokens.
- The numbers are for this GPU pair. Other SM120 cards or a different
  tensor-parallel size are untested.

After boot, the scheduler may log `Triton kernel ... device-loaded after serving started`
for a handful of small bookkeeping kernels (route packing, kpool layout and tail scatter,
slot copy). Those are alignment-specialized cubin variants loading from the persistent cache
in milliseconds, not compiles, and they carry no memory risk; they are expected under
traffic. A load that takes seconds would be a compile and is worth reporting.

## Carried upstream changes

The candidate is official SGLang and FlashInfer `main` plus checksummed patches
(`patches/`, pinned in `stack.lock.json`). It carries each listed open PR from
the exact head below and records a merged dependency where its code now comes
from `main`. General follow-up work that is not yet filed upstream is identified
explicitly as downstream pending upstream rather than presented as upstream
provenance.

SGLang (base `main` `c1b4d535d7`, 2026-09-03):

| PR | State and head | What it carries |
|---|---|---|
| [sgl-project/sglang#36507](https://github.com/sgl-project/sglang/pull/36507) | Open, `4761b54153` | GLM-5.3-Flash series; its 26 substantive commits are unchanged from the prior head and include issue #37548's first-request multimodal NextN embedding fix |
| [sgl-project/sglang#36904](https://github.com/sgl-project/sglang/pull/36904) | Open, `436a89b06f` | TileLang fp8_e4m3 KV cache on CUDA (raw layout) |
| [sgl-project/sglang#36661](https://github.com/sgl-project/sglang/pull/36661) | Open, `cc78c41a14` | overlap batch snapshot lifetime tied to result completion |
| [sgl-project/sglang#36696](https://github.com/sgl-project/sglang/pull/36696) | Open, `1ff8934369` | mamba radix cache split-node key fix |
| [sgl-project/sglang#36821](https://github.com/sgl-project/sglang/pull/36821) | Open, `948bfdd37b` | ReplaySSM ring-write in the fused KDA chain-verify kernel |
| [sgl-project/sglang#37168](https://github.com/sgl-project/sglang/pull/37168) | Open, `1c5d5cfa29` | full CUDA-graph capture owners (MHC and DSA tensors replayed by captured graphs) |
| [sgl-project/sglang#37169](https://github.com/sgl-project/sglang/pull/37169) | Open, `683154b56d` | opt-in allocator-history forensics snapshots |
| [sgl-project/sglang#37534](https://github.com/sgl-project/sglang/pull/37534) | Open, `c9853eb19b` | HiCache host pool rows sized to packed DSA KV rows |
| [sgl-project/sglang#37535](https://github.com/sgl-project/sglang/pull/37535) | Open, `27e648e690` | opt-in token-blocked KDA extend (`SGLANG_KDA_EXTEND_BLOCK_TOKENS`) |
| [sgl-project/sglang#37536](https://github.com/sgl-project/sglang/pull/37536) | Open, `17d1707234` | raw multimodal features released from the device after encoding |
| [sgl-project/sglang#37537](https://github.com/sgl-project/sglang/pull/37537) | Open, `7e27c6123e` | `--mm-preprocessing-device` for the base visual preprocessing path |
| [sgl-project/sglang#37538](https://github.com/sgl-project/sglang/pull/37538) | Open, `bcff46c9a8` | env-gated extend memory profiler |
| [sgl-project/sglang#37539](https://github.com/sgl-project/sglang/pull/37539) | Open, `00cc3e8dc9` | GLM-5-Next vision-tower attention and compiled-activation precompile at startup |
| [sgl-project/sglang#37625](https://github.com/sgl-project/sglang/pull/37625) | Open, `7ad39a0665` | oversized DSA radix-bin recovery across legacy and DeepSeek-V4 AOT/JIT variants, co-authored by @bold84 |
| [sgl-project/sglang#37541](https://github.com/sgl-project/sglang/pull/37541) | Open, `ade49acbb3` | opt-in `serving_coverage` request warmup |
| [sgl-project/sglang#37612](https://github.com/sgl-project/sglang/pull/37612) | Open, `595d6b45b7` | prefill admission re-evaluated every round on hybrid SSM radix caches (the v0.1.2 C4 fix) |
| [sgl-project/sglang#37619](https://github.com/sgl-project/sglang/pull/37619) | Open, `ed46bc9c3c` | unfinished-request mamba checkpoint skipped when the pool is exhausted (the v0.1.3 crash fix) |
| [sgl-project/sglang#37744](https://github.com/sgl-project/sglang/pull/37744) | Open, `ebe935c116` | qkvbfg fusion gated by each source layer's resolved quantization method |
| [sgl-project/sglang#37375](https://github.com/sgl-project/sglang/pull/37375) | Open, `b6478a7400` | mHC pipeline stages hand off the flattened hidden state without a separate residual |

FlashInfer (base `main` `9f5051736e`, 2026-09-03):

| PR | State and head | What it carries |
|---|---|---|
| [flashinfer-ai/flashinfer#4802](https://github.com/flashinfer-ai/flashinfer/pull/4802) | Merged, `453aa7c729` | native SM120 sparse-MLA runner with GLM NoPE rows; supplied directly by current `main` |
| [flashinfer-ai/flashinfer#4687](https://github.com/flashinfer-ai/flashinfer/pull/4687) | Open, `b75d6bfff7` | W4A16 large weight bank addressing |
| [flashinfer-ai/flashinfer#4827](https://github.com/flashinfer-ai/flashinfer/pull/4827) | Open, `21fb169ff2` | SM12x MoE workspaces kept alive while captured CUDA graphs reference them |

Downstream-only in the patches (no upstream PR): the SM120 W4A16 MoE and
NoPE-row integration, the ModelOpt E4M3-K32 W4A16 weight preparation, the
adaptive-MTP chain-buffer pinning, and the Triton late-load diagnostic
armed after the serving warmup. Pending-upstream follow-ups cover exact
GLM video CPU placement and EPD decoder lifecycle, DP video sampling parity,
collision-safe media ordering, and stricter NextN multimodal failure handling.
Already merged upstream and in the base:
[sgl-project/sglang#37317](https://github.com/sgl-project/sglang/pull/37317),
[#36958](https://github.com/sgl-project/sglang/pull/36958),
[#36798](https://github.com/sgl-project/sglang/pull/36798), and since
2026-09-02 [#37477](https://github.com/sgl-project/sglang/pull/37477) (the
GLM-5.3-Flash kernels, ported from #36507).

## Releases

`v0.2.1-rc.6` (2026-09-03) is the current unbuilt source candidate. It rebases
the complete integration onto SGLang main `c1b4d535d7` and FlashInfer main
`9f5051736e`, fetched immediately before patch generation. All 22 tracked PRs
were rechecked: FlashInfer #4802 is now merged and supplied by `main`; the 19
other previously unchanged heads remain unchanged; SGLang #36507 moved only by
merging `main` plus an AMD change and its immediate revert; and #37625 advances
to final head `7ad39a0665`. The top-k portion of the resulting integration has
the final PR's exact stable patch ID. The scheduler rebase preserves current
main's tiered cache-admission accounting alongside the downstream Mamba-slot
gate. A focused CPU regression covers free, evictable, and int8-checkpoint
static-Mamba slot accounting. The cache schema advances to `v61`. This
candidate is not built or qualified and inherits no claims from earlier
candidates.

`v0.2.1-rc.5` (2026-09-03) is the latest internally qualified candidate. It rebases
the complete v0.2.1 integration without conflict onto SGLang main
`05dbe64dff` and FlashInfer main `7a3c04f015`, rechecks every carried PR head,
retains the allocation-free exact fallback for exact-capacity DSA top-k bins,
and strengthens the boundary regression with negative keys and already-emitted
higher values. It advances the cache schema to `v60` and was built internally
at `sha256:289e83c983fb951ed5265de80cca3dc0412cc0cd43cf2190296a7f8190c38f69`.
The exact image passed the source, isolated GPU, first-boot, crash, full quality,
and analyzer-validated engine gates on quasar. The primary qualification
repository holds the measured receipts. It was not promoted or published.
`v0.2.1-rc.4` (2026-09-03) was built internally at
`sha256:7a64f03935c0d862cd1352ee276016a45eb627ed2e1e3f2576353fe933a95e0a`.
It reached Ready with zero restarts, served the first sampled thinking-mode
chat, and passed 196 focused CPU tests plus 75 subtests with the GPUs hidden
from pytest. It was superseded before GPU-kernel or quality qualification when
both upstream mains advanced and was not promoted or published.
`v0.2.1-rc.3` (2026-09-03) was built internally at
`sha256:5f362c4c6621ef2e420870b749cb123255ce214a51c77e2f24695cebfe614c8a`
but its exact-image GPU gate found an unfilled top-k slot at the shared-stash
capacity boundary. It was rejected and was not promoted or published.
`v0.2.1-rc.2` (2026-09-03) was built and qualified internally at
`sha256:39bbf5b178ed90cfabc2f76636c3e707e5bafcc1861be3665c8777b3433dff4a`.
It was not promoted or published and is superseded by the current-main rebase.
The complete receipt remains in the primary qualification repository under
`evidence/v0.2.1-rc.2-validation-20260903.md`. `v0.2.0` (2026-09-03) is the
current published release, a digest-identical promotion
of its rc.1 candidate: the v0.1.4 bases and serving configuration with four
decode-path changes (PCIe IPC all-reduce wired in, fused KDA gate
projections under the quantized config, upstream's experimental device-side
DSA kpool metadata, an FP8 lm_head shared by target and draft), +19% engine step rate at
concurrency 1 on the fixed-acceptance probe and +9% on the four-request
gate, same pool, same GSM8K score (see "What to expect"). `v0.1.4`
(2026-09-02) was a digest-identical promotion
of its rc.2 candidate: the same serving configuration as v0.1.3 on refreshed
upstream bases (SGLang main f8cbf000f4 with the merged #37477 kernel port,
FlashInfer main c92227fad3 with #4802 round 2), with every carried pull
request applied from its current PR head. `v0.1.3` (2026-09-02) was the
crash-free configuration: pool and context 450,560 tokens, and a scheduler
that skips a recurrent-state checkpoint instead of asserting when the
mamba pool is exhausted (see the limits above and `BENCHMARKS.md`).
`v0.1.2` (2026-09-02) fixed the scheduler
latch that kept a fourth concurrent request queued until another finished,
so four requests decode together. `v0.1.1` (2026-09-02) fixed the Xid 31
fault in the DSA top-k kernel under long prefill (reported by @bold84 and
@sousekd, fixed by @bold84), large-image OOMs and first-request warmup, and
moved to the 2026-09-02 upstream mains. Details and every candidate build
are in [`CHANGELOG.md`](CHANGELOG.md).

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
  -t sglang-glm53-flash-sm120:v0.2.1-rc.6 .
```

The release workflow refuses to overwrite an existing SemVer candidate tag.

## License

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Upstream SGLang,
FlashInfer, ModelOpt and GLM-5.3-Flash retain their own licenses.
