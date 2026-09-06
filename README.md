# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs

A ready-to-run SGLang image and a matching quantized checkpoint for serving
GLM-5.3-Flash on two NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) GPUs over
PCIe. No NVLink, no source build, no patching: download the checkpoint, run
one command, and you have an OpenAI-compatible server with a 450,560-token
context, four concurrent requests, speculative decoding, vision input,
reasoning and tool calling.

| | |
|---|---|
| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.2.1` |
| Checkpoint | [`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO) on Hugging Face |
| Hardware | 2x RTX PRO 6000 Blackwell (SM120), tensor parallel 2, PCIe |
| Candidate | `sglang-glm53-flash-sm120:v0.3.1-rc.1` (built internally; not qualified) |

`v0.3.1-rc.1` tests a smaller W4A16 MoE route-prefix kernel for concurrent decoding. It retains the `v0.3.0` SGLang runtime, model, quantization and serving settings. The changed FlashInfer source passed isolated GPU correctness, graph replay and sanitizer checks. The internal candidate at `sha256:5c2c6fb8f5616d3b451d45d944f56248f0e81c6cc403b80aa8c296f8391b5e2d` completed sampled first-boot, image/cold-cohort and matched application checks. Concurrent decode improved, while the single-request decrease remains unresolved. [Measurements and limits](https://git.home.corenode.com/homelab/sglang-glm53-flash-sm120/src/branch/main/evidence/v0.3.1-rc.1-route-pack-20260906/README.md) are recorded in the primary project. Full exact-candidate qualification and promotion have not been performed.

The preceding `v0.3.0-rc.4` refreshes SGLang to main `febb360519` and retains current
FlashInfer main `6c14bbd5ff` and the audited carry heads. It preserves the
hybrid HiCache index fix, removes obsolete encoder cleanup, moves receive
timeouts before rank consensus and synchronizes host-memory budget readings.
The internal image is built and qualified at `sha256:86dc493a8d64df2f1beae4b15e3224cad7f6b9fb741183affb18ff5d1882c0c3`. Its exact-image GPU, sampled first-boot, image/cold-cohort, full GSM8K, standardized C1-C4/prefill, long-context and forced-host restoration checks completed. Public publication is pending. `v0.3.0-rc.3` completed its serving gates but
is withheld from promotion because the subsequent lifecycle corrections are
absent from that immutable image. Validation and measured results belong in
the primary qualification repository; no previous release qualifies new source.

The published `v0.2.1` image has a reproduced long-prefix HiCache corruption
defect. Keep HiCache disabled on that image while the replacement is qualified.

The current published stable image is `v0.2.1`, promoted without a rebuild from
`v0.2.1-rc.8`. The internal candidate and stable tag resolve to
`sha256:9bc64968dcf3b43b974ab95189ea0f208d2d60cfda9203d0b59942470451578e`;
the independently built ghcr candidate and stable tag resolve to
`sha256:e292b3677ac8085d087fb2503fb8b42c143abe1d2739c360ee776a669f53b424`.
The internal candidate's source, exact-image GPU, first-boot, crash, full
quality, long-context, and standardized engine gates passed. The primary
qualification repository holds the measured results and receipts.
`v0.2.1-rc.7` was rejected by its
exact-image GPU gate because one new test violated the kernel's row-stride
precondition. `v0.2.1-rc.5` was built and qualified internally on quasar at
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
- Optional: 64 GB for the TP2 HiCache pools at 32 GB per rank, plus host RAM
  for the model-loading and serving processes. HiCache is off by default in
  the launcher and must remain disabled on the published `v0.2.1` image.

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
   export IMAGE=ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.2.1
   export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v63
   ./examples/serve-glm53-flash.sh
   ```

   Keep HiCache disabled on the published `v0.2.1` image because of the
   reproduced host-restore defect described above.
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

Status checked 2026-09-06 after #36507 merged into `main`. This table describes the source carried by internally qualified `v0.3.0-rc.4`, promoted digest-identically to internal `v0.3.0`. Context, API and test-fixture adaptations are retained in the integration patch. PR refiling changes the upstream review destination; it does not change the immutable image or its source pins. Public publication remains pending.

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

Downstream work still awaiting submission: SM120 MoE/NoPE integration and defaults, ModelOpt E4M3-K32 preparation, static Mamba admission/accounting, adaptive-MTP chain-buffer lifetime, GLM video/DP/media-ordering and stricter NEXTN multimodal handling, PCIe IPC all-reduce wiring, mixed-precision KDA gate fusion beyond #37744, optional FP8 lm_head, recurrent-kernel tuning, and additional diagnostics. FlashInfer #4802's merge removes that dependency blocker but does not upstream the SGLang integration. Not every retained local change has an upstream PR yet.

## Releases

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

`v0.2.1` (2026-09-04) is the current published release, promoted without a
rebuild from `v0.2.1-rc.8` in both registries. The internal candidate and
stable tag resolve to
`sha256:9bc64968dcf3b43b974ab95189ea0f208d2d60cfda9203d0b59942470451578e`;
the independently built ghcr candidate and stable tag resolve to
`sha256:e292b3677ac8085d087fb2503fb8b42c143abe1d2739c360ee776a669f53b424`.
The internal candidate passed its source, exact-image GPU, first-boot, crash,
full quality, long-context, and standardized C1-C4 engine gates. The primary
qualification repository holds the measured results and exact-candidate
receipts.

`v0.2.1-rc.8` (2026-09-04) is the current internally qualified candidate at
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
