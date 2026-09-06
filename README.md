# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs

A ready-to-run SGLang image and a matching quantized checkpoint for serving
GLM-5.3-Flash on two NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) GPUs over
PCIe. No NVLink, no source build, no patching: download the checkpoint, run
one command, and you have an OpenAI-compatible server with a 450,560-token
context, four concurrent requests, speculative decoding, vision input,
reasoning and tool calling.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.3.1` |
| Internal image | `git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.3.1` |
| Checkpoint | [`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) on Hugging Face |
| Hardware | 2x RTX PRO 6000 Blackwell (SM120), tensor parallel 2, PCIe |
| Qualified internal candidate | `git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.3.1-rc.1` |

The current internal stable image is `v0.3.1`. The current published stable image is `v0.3.1`. Each stable tag was promoted without rebuilding its `v0.3.1-rc.1` candidate. The qualified internal digest is `sha256:5c2c6fb8f5616d3b451d45d944f56248f0e81c6cc403b80aa8c296f8391b5e2d`; the independently built public digest is `sha256:6c5b7e6701fb64e1890f067cbad5bff2d5c979fb3952f520862a7dfb858cdea3`. Both builds use identical pinned image inputs. Hardware measurements apply to the internal digest; the public build has separate source and build provenance.

`v0.3.1` replaces small-batch W4A16 MoE route-prefix comparison tiles with a histogram and inverse-prefix lookup to avoid expert-by-route and expert-by-block comparisons. It includes the hybrid DSA HiCache index-restoration and lifecycle corrections from the v0.3.0 integration. The model, quantization and serving settings are retained. SGLang is pinned to main `febb360519` with GLM support carried from the pre-merge #36507 branch; FlashInfer is pinned to main `6c14bbd5ff` plus the changes listed below.

The exact internal candidate passed GPU correctness/sanitizer tests, startup and image/cold-C4 acceptance, full GSM8K, all 24 engine cell analyzers, long-context controls and forced-host restoration. The engine summary's adaptive-gauge handling was corrected and independently reviewed without rerunning or changing any measured cell. [Measurements and retained limitations](BENCHMARKS.md#v031-qualification-2026-09-06) and [release receipts](evidence/v0.3.1/README.md) are included here.

The older `v0.2.1` image has a reproduced long-prefix HiCache corruption defect. Keep HiCache disabled if continuing to use that version.

## Requirements

- Linux x86_64 with a CUDA 13 capable driver, Docker and the NVIDIA
  Container Toolkit.
- Two visible SM120 GPUs. Other GPU pairs are untested.
- About 170 GB of disk for the checkpoint and a few GB for the kernel cache.
- Optional: 64 GB for the TP2 HiCache pools at 32 GB per rank, plus host RAM
  for the model-loading and serving processes. HiCache is off by default in
  the launcher; the qualification profile enables it at 32 GB per rank.

## Run it

1. Download the checkpoint.

   ```bash
   pip install -U huggingface_hub
   export MODEL_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
   HF_XET_HIGH_PERFORMANCE=1 hf download ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO --local-dir "$MODEL_DIR"
   ```

2. Start the server with the launcher from this repository. Enable the optional
   HiCache settings below to match the qualification profile.

   ```bash
   git clone https://github.com/ormandj/sglang-glm53-flash-sm120
   cd sglang-glm53-flash-sm120
   export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.3.1
   export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v68
   ./examples/serve-glm53-flash.sh
   ```

   To enable the qualified 32 GB-per-rank host-cache profile, set
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

- `--cuda-graph-bs-decode` lists every batch size up to
  `--max-running-requests`, so no decode batch replays through a padded
  graph. On builds before the 2026-09-02 upstream base, padded replay
  corrupted outputs on this hybrid model (GSM8K 68% against 90% with graphs
  at batch 1 and 4 only); on v0.1.4 the same test scores 98.3% on 300
  questions, so the exact list is kept as a precaution, not a requirement.
- `--max-mamba-cache-size` is recurrent-state slots, not KV cache. Each live
  request uses four to five, so the launcher ships 28 for four requests.

## What to expect

Measured on the qualified internal `v0.3.1-rc.1` digest above, using two RTX PRO 6000 Blackwell Max-Q GPUs at 300 W and the launcher profile with 32 GB of HiCache per rank. The independently built public image was not separately tested on the serving GPUs.

| Measurement | Result |
|---|---|
| Fixed-acceptance C1, 1k / 19k context | 67.41 / 66.99 forwards/s; six repetitions each |
| Engine C1 / C2 / C3 / C4 | 62.61 / 48.91 / 38.75 / 34.24 mean forwards/s; five repetitions each |
| Cold prefill, 8k / 32k / 64k / 128k | 5,291 / 5,944 / 5,970 / 5,935 prompt tok/s |
| Full GSM8K | 1,280/1,319 with unchanged GLM-aware grading; 1,181/1,319 with pinned AIPerf |
| 73k / 400k quality controls | 143/150 / 145/150 |
| Long-context continuation | 767/767 next-token choices agreed |
| Host restoration | Five tool-decision cycles and seven ordered markers at 408k passed |
| Final memory acceptance | 7680x4320 image, then four independent cold 4k requests; zero restarts |

Forwards/s counts engine decode steps. On the fixed-acceptance ledger probe, mean output rates were 154.9 / 154.6 tok/s at 2.30 / 2.31 accepted tokens per forward for 1k / 19k context ([receipt](evidence/v0.3.1/fixed-acceptance-summary.json)).

Engine rates use a synthetic fixed output window and depend on adaptive speculative paths; they are not application-throughput promises. Two GSM8K responses exhausted their output budget and remain in the scores. Exact-text host recall differed only in optional marker labels; the pre-existing ordered-marker oracle passed. The server's adaptive acceptance-rate gauge can mix draft widths within one reporting interval and is diagnostic, not a probability. The initial C1 decrease and the subsequent follow-up are both retained in [BENCHMARKS.md](BENCHMARKS.md).

Limits for this profile:

- A request for input logprobs spanning a long prompt can still OOM the scheduler and restart the container. Score only the continuation at the prompt boundary.
- Memory is tightly sized at `mem-fraction-static=0.99`. Prefill, vision and runtime compilation share the remaining headroom; keep `max-prefill-tokens` equal to the 4,096-token chunk size. Increasing the pool, concurrency or image budget requires new memory acceptance tests.
- Images use approximately one token per 28x28 pixels up to the checkpoint's 8,000-token limit. Larger images are resized by the processor; image tokens consume context capacity.
- Small bookkeeping kernels can log `device-loaded after serving started` for alignment-specialized variants loaded from the persistent cache in milliseconds. Such lines remain present on this release. A first request delayed by seconds of compilation needs investigation.
- Measurements cover this two-GPU SM120 profile. Other cards and tensor-parallel sizes are untested.

## Carried upstream changes

Status checked 2026-09-06 after #36507 merged into `main`. This table describes the source carried by `v0.3.1`, including the retained v0.3.0 integration. Context, API and test-fixture adaptations are retained in the integration patch. PR refiling changes the upstream review destination; it does not change the immutable image or its source pins.

Audited source bases: SGLang `main` `febb360519`, GLM #36507 `be2e63c2f1`, FlashInfer `main` `6c14bbd5ff`. Main includes #38163's AMD unified-KV revert and #36988's aborted disaggregated-prefill retirement. Image provenance is the exact pins and patch checksums in `stack.lock.json`.

| SGLang PR | Current state / carried source head | Behavior addressed |
|---|---|---|
| [#36507](https://github.com/sgl-project/sglang/pull/36507) | Merged / `be2e63c2f1` | GLM-5.3-Flash model support, including the multimodal NEXTN fix for #37548 |
| [#37980](https://github.com/sgl-project/sglang/pull/37980) | Open / `a340acbe5b` | Order speculative plan-stream work and preserve the Mamba top-k-1 fast paths |
| [#36904](https://github.com/sgl-project/sglang/pull/36904) | Closed after base deletion / `436a89b06f` | Make raw-layout FP8 KV usable with CUDA TileLang DSA |
| [#36661](https://github.com/sgl-project/sglang/pull/36661) | Open / `cc78c41a14` | Keep overlap batch snapshots alive until result processing completes |
| [#36696](https://github.com/sgl-project/sglang/pull/36696) | Open / `1ff8934369` | Register split Mamba cache nodes under their own keys |
| [#36821](https://github.com/sgl-project/sglang/pull/36821) | Open / `948bfdd37b` | Correct recurrent-state ring writes during fused KDA chain verification |
| [#37168](https://github.com/sgl-project/sglang/pull/37168) | Open / `1c5d5cfa29` | Prevent full CUDA graphs from replaying through freed MHC/DSA tensor storage |
| [#37169](https://github.com/sgl-project/sglang/pull/37169) | Open / `683154b56d` | Capture allocator history to investigate memory corruption; opt-in diagnostic |
| [#37534](https://github.com/sgl-project/sglang/pull/37534) | Open / `c9853eb19b` | Match host/device packed DSA row sizes during HiCache transfers; adapted dependency of #38212 |
| [#37535](https://github.com/sgl-project/sglang/pull/37535) | Open / `27e648e690` | Bound each KDA prefill call's workspace with opt-in internal token blocks |
| [#37536](https://github.com/sgl-project/sglang/pull/37536) | Open / `17d1707234` | Release raw multimodal device tensors before language-model execution |
| [#37537](https://github.com/sgl-project/sglang/pull/37537) | Open / `7e27c6123e` | Allow CPU preprocessing to avoid the base visual path's extra CUDA context |
| [#37538](https://github.com/sgl-project/sglang/pull/37538) | Open / `bcff46c9a8` | Attribute prefill allocation growth to instrumented phases; opt-in diagnostic |
| [#38214](https://github.com/sgl-project/sglang/pull/38214) | Open successor; carries #37539 `00cc3e8dc9` | Precompile vision attention and MLP activations before KV allocation |
| [#37541](https://github.com/sgl-project/sglang/pull/37541) | Open / `ade49acbb3` | Warm up sampled, batched and multimodal serving paths before first traffic |
| [#37612](https://github.com/sgl-project/sglang/pull/37612) | Open / `595d6b45b7` | Retry queued prefills after a Mamba-aware cache admission failure |
| [#37619](https://github.com/sgl-project/sglang/pull/37619) | Open / `ed46bc9c3c` | Skip optional unfinished checkpoints when Mamba slots run out instead of crashing |
| [#37625](https://github.com/sgl-project/sglang/pull/37625) | Open / `4302a2b719` | Correct top-k selection for exact-capacity/oversized bins across AOT/JIT paths; co-authored by @bold84 |
| [#37744](https://github.com/sgl-project/sglang/pull/37744) | Closed after base deletion / `ebe935c116` | Enable KDA projection fusion when the relevant layers are unquantized |
| [#37375](https://github.com/sgl-project/sglang/pull/37375) | Closed after base deletion / `b6478a7400` | Avoid the GLM pipeline-parallel `residual` KeyError by using the mHC handoff contract |
| [#38157](https://github.com/sgl-project/sglang/pull/38157) | Open / `8128658833` | Synchronize TP host-memory readings before pool allocation to avoid a false startup OOM |
| [#38212](https://github.com/sgl-project/sglang/pull/38212) | Open successor; carries #38161 `66f6da3a21` | Restore missing target/draft attention indexes after HiCache loadback and separate divergent compressed prefixes |
| [#38213](https://github.com/sgl-project/sglang/pull/38213) | Open successor; carries #38162 `efd2a02d03` | Reduce repeated attention-metadata setup during speculative decoding; opt-in optimization |
| [#38164](https://github.com/sgl-project/sglang/pull/38164) | Open / `7c2a647804` | Expire stuck preallocation waits before rank consensus; correct prefill and health-check queue handling |

#36507 merged on 2026-09-06. Its deleted base branch auto-closed #38161, #38162 and #37539; their unchanged patches are now #38212, #38213 and #38214 against `main` `938dc5621d`. The corresponding carried source heads above remain the original heads. All open SGLang PRs listed here now target `main`, including #37980. #36904, #37375 and #37744 are closed without a PR merge; closure alone does not establish that their fixes reached main. In-graph/preallocated metadata experiments removed by #38071 are not restored by #38213. #38164 replaces the retained lifecycle follow-ups from closed #37316; obsolete encoder cleanup is excluded.

| FlashInfer PR | State / source | Behavior addressed |
|---|---|---|
| [#4802](https://github.com/flashinfer-ai/flashinfer/pull/4802) | Merged / merge commit `453aa7c729` | Native SM120 sparse-MLA with GLM NoPE rows; supplied by FlashInfer main |
| [#4687](https://github.com/flashinfer-ai/flashinfer/pull/4687) | Open / head `b75d6bfff7` | Correct addressing of large W4A16 expert weight banks |
| [#4827](https://github.com/flashinfer-ai/flashinfer/pull/4827) | Open / head `21fb169ff2` | Keep graph-referenced MoE workspaces alive after cache growth or clearing |

Already merged dependencies include SGLang #37317, #36958, #36798 and #37477. The carried GLM source also includes #37250, #36884, #36885 and the #37548 fix at `cdfc224b0e`. GLM support has since merged through #36507 at `97c6978369`; that merge commit is upstream status, not this image's source pin.

Downstream work still awaiting submission: the small W4A16 route-prefix histogram/inverse-prefix change (reviewed FlashInfer fork branch `pr/sm12x-route-prefix-histogram`, PR prepared), SM120 MoE/NoPE integration and defaults, ModelOpt E4M3-K32 preparation, static Mamba admission/accounting, adaptive-MTP chain-buffer lifetime, GLM video/DP/media-ordering and stricter NEXTN multimodal handling, PCIe IPC all-reduce wiring, mixed-precision KDA gate fusion beyond #37744, optional FP8 lm_head, recurrent-kernel tuning, and additional diagnostics. FlashInfer #4802's merge removes that dependency blocker but does not upstream the SGLang integration. Not every retained local change has an upstream PR yet.

## Releases

`v0.3.1` (2026-09-06) is the current stable image in both registries. It promotes each registry's `v0.3.1-rc.1` candidate digest-identically. See the [current changelog](CHANGELOG.md) for behavior changes, measured validation and limitations.

`v0.3.0` was released internally from qualified `v0.3.0-rc.4` and is superseded by `v0.3.1`. It was not published to GHCR.

`v0.3.0-rc.3` (2026-09-05) is built internally at
`sha256:5965a1d2beb0ca824a74ce2e95df49088d00e5b409b3a862ee082662f505a4b0`.
Its validation gates completed, but promotion is withheld after final source
review. Corrected lifecycle source and a further base refresh require a new
candidate and full exact-image qualification. It uses compiled-cache namespace
`v66`. The preceding
`v0.3.0-rc.1` metadata identifies the old-base HiCache diagnostic, published
only under its immutable test-commit image tag, not as a SemVer release.

`v0.3.0-rc.2` built internally at
`sha256:f66f49b75cc960559ec5eaa5c80d7f9743498e03743d47f5d0683417991d2124`
but was rejected by its exact-image test gate: the reachable-value mutation
probe correctly detected the injected mutation, while its assertion expected
an older diagnostic string. `v0.3.0-rc.3` corrects only that test expectation;
the runtime is unchanged. The previous candidate was never activated for
serving or promoted.

`v0.2.1` (2026-09-04) was the preceding published stable release, promoted without a
rebuild from `v0.2.1-rc.8` in both registries. The internal candidate and
stable tag resolve to
`sha256:9bc64968dcf3b43b974ab95189ea0f208d2d60cfda9203d0b59942470451578e`;
the independently built ghcr candidate and stable tag resolve to
`sha256:e292b3677ac8085d087fb2503fb8b42c143abe1d2739c360ee776a669f53b424`.
The internal candidate passed its source, exact-image GPU, first-boot, crash,
full quality, long-context, and standardized C1-C4 engine gates. The primary
qualification repository holds the measured results and exact-candidate
receipts.

`v0.2.1-rc.8` (2026-09-04) was internally qualified at
`sha256:9bc64968dcf3b43b974ab95189ea0f208d2d60cfda9203d0b59942470451578e`.
It retains the rc.7 kernel implementation unchanged and pads the new exact-tail
test's physical row stride to the V2 kernel's 16-byte vector-load contract
while preserving the 5,047-element logical row. Its source, exact-image GPU,
first-boot, crash, full quality, long-context, and standardized C1-C4 engine
gates passed. The cache schema advances to `v63`. The primary qualification
repository holds the measured results and exact-candidate receipts. The
candidate was promoted without a rebuild as `v0.2.1` in both registries.

`v0.2.1-rc.7` (2026-09-04) was built internally at
`sha256:e7945f72cf038ad83a034f2ba1f0d8abf23e977b7576dad7063331c6cb91e16c`
and rejected by its exact-image GPU gate because the new exact-tail regression
used a physical row stride outside the kernel contract. Its kpool suite and the
other selected V2 tests passed before that harness failure; the source kernel
change itself is retained in rc.8. It otherwise retained
the official bases and reviewed #37625 head from rc.6, then added a downstream
overflow follow-up that discovers the exact cutoff before writing output and
performs one final disjoint write of strict winners and cutoff ties. The
DeepSeek-V4 path reuses its bounded captured-candidate buffer when every value
in the oversized bin has one exact key. Focused coverage now exercises cutoff
separation at each variable key byte, exact ties after an almost-full strict
prefix, positive and negative zero, and the legacy AOT/JIT copies. The cache
schema advanced to `v62`. It was not qualified and inherits no claims from
earlier candidates.

`v0.2.1-rc.6` (2026-09-03) was superseded before build or qualification. It rebases
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

`v0.2.1-rc.5` (2026-09-03) was an internally qualified candidate. It rebases
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
`evidence/v0.2.1-rc.2-validation-20260903.md`. `v0.2.0` (2026-09-03) was a
digest-identical promotion
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
  -t sglang-glm53-flash-sm120:v0.3.1-rc.1 .
```

The release workflow refuses to overwrite an existing SemVer candidate tag.

## License

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Upstream SGLang,
FlashInfer, ModelOpt and GLM-5.3-Flash retain their own licenses.
