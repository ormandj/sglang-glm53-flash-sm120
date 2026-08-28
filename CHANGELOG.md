# Changelog

## v0.1.0-rc.13

No-RoPE MLA reserved-slot correctness candidate. This candidate is not
qualified.

- Fix GLM's dedicated no-RoPE MLA scatter to apply the same
  `reserved_skip_index` and DCP ownership mapping as the ordinary MLA writer.
  The previous branch wrote directly to physical slot 0, allowing CUDA-graph
  padding to poison a cache row reserved for padding reads.
- Add a source contract and exact-GPU regression. The regression reproduces
  `slot0_changed=True` on v0.1.0-rc.12 while its positive-slot control write
  succeeds.
- Advance cache schema to `v9` because the effective KV writer changed.
- Retain the v0.1.0-rc.12 2,051-entry unfused transform, refreshed FlashInfer
  pin, full 524,288-token target pool, vision configuration, and fixed native
  MTP profile.

## v0.1.0-rc.12

Unfused KPool contract diagnostic candidate. This candidate is not qualified.

- Fix the generic DSA index transform to preserve GLM-5.3's exact 2,051-entry
  table (2,048 history entries plus three live KPool tails) in both decode and
  prefill. The tuned 2,048-entry decode kernel remains unchanged, and every
  other width is still rejected.
- Add a source contract plus an exact-GPU comparison against the torch
  reference, including a padded gap before the three live tail entries.
- Refresh FlashInfer main to exact commit `93f4f264…`, tree `7e9829d1…`. The
  intervening changes do not touch sparse-MLA sources, and the isolated
  GLM_NEXT_NOPE patch still applies with zero fuzz.
- Advance cache schema to `v8` because the effective SGLang transform and
  FlashInfer tree changed.
- Retain v0.1.0-rc.11's CPU image preprocessing, fixed five-step native MTP,
  concurrency-one memory envelope, and default PyTorch allocator.
- Exact-hardware startup succeeded with a 524,288-token FP8 target pool and
  4.06 GB/rank available after target decode graph capture. Target-only
  inference remained corrupt, returning a repeated `742` pattern for a
  deterministic marker request. The candidate was scaled back to zero and was
  not qualified.

## v0.1.0-rc.11

Exact GLM-5.3 no-RoPE sparse-MLA candidate. This candidate is not qualified.

- Retain the isolated 528-byte no-RoPE FlashInfer trait, 2,176-column padded
  KPool dispatch, TP=2 specialization, and build-time CUDA/numerical gates
  introduced for v0.1.0-rc.10.
- Correct the fail-closed final `flash_mla_sm120.py` postimage from a value not
  produced by the exact vendor base to the observed vendor-base result,
  `8ccc7bb2…`. All other exact preimages and postimages matched.
- Preserve cache schema `v7`: kernel and adapter bytes are unchanged from the
  intended v0.1.0-rc.10 implementation.
- Keep the direct-run launcher on PyTorch's default allocator. Expandable
  segments reproduce upstream SGLang issue #17642 in the two-GPU PCIe
  CustomAllreduce CUDA-graph capture path.
- Keep image preprocessing on CPU so the vision warmup does not contend with
  the capacity-first KV pool and native-MTP CUDA graphs for residual VRAM.
- Use fixed five-step native MTP at concurrency one. The default adaptive
  controller's extra 1/3/7-step graph families leave only about 202 MiB free,
  below the vision encoder warmup's observed 256-MiB allocation.

## v0.1.0-rc.10

Exact GLM-5.3 no-RoPE sparse-MLA attempt. No image was published: the build
failed closed when the final adapter postimage did not match the exact vendor
base. The failure is preserved in
[`evidence/v0.1.0-rc.10-build-failure-20260828.txt`](evidence/v0.1.0-rc.10-build-failure-20260828.txt).

- Correct the cache-layout diagnosis: GLM-5.3 uses SGLang's generic 528-byte
  cache (512 E4M3 latent bytes plus four arbitrary FP32 scales), not the
  584-byte DSv4 footer cache assumed by v0.1.0-rc.9.
- Add an isolated FlashInfer SM120 model trait for the 512-dimensional no-RoPE
  query/cache geometry while leaving DSv3.2, GLM-NSA, and DSv4 unchanged.
- Preserve the fixed 2,048-entry DSA table plus three live KPool entries by
  padding 2,051 columns with `-1` to 2,176 (34 complete 64-entry tiles).
- Add only the TP=2 production dispatch: 32 query heads and 2,176 candidates,
  for both split-K decode and prefill.
- Add build-time adapter contracts, build-time NVCC compilation, and exact-SM120
  numerical decode/prefill tests with a padding gap before live tail entries.
- Advance cache schema to `v7` because the effective SGLang adapter and
  FlashInfer kernels changed.

## v0.1.0-rc.9

GLM-5.3 DSv4/KPool sparse-MLA correctness candidate. This candidate is not
qualified and was invalidated by the rc.10 cache-path audit: its 584-byte
branch was dead for the model's actual 528-byte cache.

- Preserve the model's fixed 2,048-entry DSA selection table plus all three
  live `index_kpool=4` tail entries. v0.1.0-rc.8 loaded both target and native
  MTP weights, then failed closed because the inherited backend guard rejected
  KPool tails before warmup completed.
- Route the 584-byte DSv4 footer cache through FlashInfer's supported
  dual-segment SM120 API: 128 primary entries plus 1,923 extra entries, with
  both segments pointing at the same physical KV cache.
- Use full segment lengths so `-1` padding in the base table cannot mask valid
  KPool tail entries that follow it. Preserve the unrelated 656-byte GLM-NSA
  adapter unchanged and open the backend guard only for the corrected 584-byte
  path.
- Pass one-token decode and 128-token prefill numerical regressions against the
  dequantized reference on the exact RTX PRO 6000 Blackwell hardware, including
  a 512-entry padding gap before the live tail.
- Refresh FlashInfer main to exact commit `950376c4…`, tree `d44c17d4…`; the
  intervening changes affect BF16 cuBLASLt work and do not change the tested
  sparse-MLA sources.
- Advance cache schema to `v6` because the effective SGLang DSA adapter and
  FlashInfer tree changed.

## v0.1.0-rc.8

SM120 source-JIT and MXFP4 post-loader correctness candidate. This candidate
is not qualified.

- Repair the SM120 MXFP4 patch to read `moe_runner_config` from the
  `Mxfp4MoEMethod` that owns it. The prior patch incorrectly read the layer;
  the exact vendor GPU suite passed its GLM numerical case but failed the
  GPT-OSS compatibility case with `AttributeError`.
- Extend the build-time semantic contract so that invalid configuration
  ownership cannot silently return.
- Remove the generic `flashinfer-cubin` package. The v0.1.0-rc.6 build spent
  24m41s enumerating 48,391 cross-architecture artifacts and then failed
  all-or-nothing after an exhausted download retry; no image was published.
- Retain the exact FlashInfer source pin, disable runtime artifact downloads,
  and use source JIT. On exact SM120 hardware, the GLM MXFP4 numerical path and
  four production-shaped sparse-MLA decode/prefill cases pass without either
  `flashinfer-cubin` or `flashinfer-jit-cache`.
- Retain cache schema `v5`; the compiled kernel sources are unchanged. The
  repaired configuration lookup runs during Python weight post-processing.

The build failure is preserved in
[`evidence/v0.1.0-rc.6-build-failure-20260828.txt`](evidence/v0.1.0-rc.6-build-failure-20260828.txt).
Exact hardware controls are preserved in
[`evidence/v0.1.0-rc.5-no-cubin-sm120-hardware-20260828.txt`](evidence/v0.1.0-rc.5-no-cubin-sm120-hardware-20260828.txt).

## v0.1.0-rc.7

GLM47 streaming tool-parser correctness candidate. This candidate is not
qualified.

- Backport only the GLM47-specific part of merged SGLang commit `0665102c…`
  (PR #36626).
- Close the outer streaming JSON object after a nested object argument; the
  exact vendor parser otherwise emits one fewer closing brace and malformed
  tool-call arguments.
- Resolve tool properties declared under legal top-level
  `anyOf`/`oneOf`/`allOf` schemas.
- Fail closed on exact vendor preimages for both modified Python files, assert
  their postimages, and run flat nested-object plus composite-schema streaming
  contracts at image build time.
- Retain cache schema `v5` because compiled SGLang and FlashInfer source is
  unchanged from v0.1.0-rc.6.

The exact vendor failure is preserved in
[`evidence/v0.1.0-rc.5-glm47-nested-tool-regression-20260828.txt`](evidence/v0.1.0-rc.5-glm47-nested-tool-regression-20260828.txt).

## v0.1.0-rc.6

Native GLM-5.3 SM120 sparse-MLA compatibility candidate. This candidate is not
qualified.

- Keep FP8 E4M3 KV while routing both DSA phases through upstream #26928's
  dedicated FlashInfer sparse-MLA kernel for SM120/SM121. Generic TRT-LLM MLA
  aborts on workstation Blackwell with `Unsupported architecture`.
- Extend only the fail-closed GLM architecture allowlist to
  `Glm5NextForConditionalGeneration` and its NextN class. The vendor guard
  otherwise rejects GLM-5.3 after both target and MTP weights load.
- Assert exact vendor preimage and postimage SHA-256 values and exercise both
  admitted classes plus the architecture, device, KV-dtype, and paired-backend
  rejection cases at image build time.
- Archive exact draft PR #36745 head `f14393b2…` after the live vendor preimage
  failed all four upstream GPU regressions: exact ties, canonical output order,
  coarse-bin overflow, and concurrent-stream determinism.
- Advance cache schema to `v5` because the effective SGLang source differs from
  v0.1.0-rc.5.

Exact v0.1.0-rc.5 failure and memory evidence is preserved in
[`evidence/v0.1.0-rc.5-sm120-startup-diagnostics-20260828.txt`](evidence/v0.1.0-rc.5-sm120-startup-diagnostics-20260828.txt).
The KPool failures are preserved separately in
[`evidence/v0.1.0-rc.5-kpool-topk-regressions-20260828.txt`](evidence/v0.1.0-rc.5-kpool-topk-regressions-20260828.txt).

## v0.1.0-rc.5

SM120 mHC/indexer fallback candidate. Superseded before qualification after the
exact-hardware diagnostic exposed the stale sparse-MLA architecture allowlist.

- Bake the same seven SM120 fallback settings that SGLang currently applies to
  `DeepseekV4ForCausalLM` into this GLM-specific image. The shared GLM
  DeepSeek/DSA hook does not inherit that architecture-specific block.
- Disable the unavailable DeepGEMM TF32 mHC prenorm path on workstation
  Blackwell. An exact-hardware v0.1.0-rc.3 diagnostic loaded both target and
  native MTP weights, then reproduced upstream issue #29738's `deep_gemm`
  `NameError` during the first warmup forward.
- Select the SM120-safe top-k, mHC, FP8 paged-MQA, and TileLang indexer settings
  as a coherent contract rather than fixing only the first observed exception.
- Record the completed, hash-verified 184,945,092,190-byte production artifact
  and advance cache schema to `v4`.

## v0.1.0-rc.4

DFlash mHC correctness and refreshed FlashInfer candidate. Superseded before
qualification after the exact-hardware diagnostic exposed the missing GLM
SM120 runtime guard.

- Apply upstream PR #36755 after #36708 so DFlash hidden-state capture accepts
  the real mHC `residual=None` contract instead of attempting to add `None` to
  the widened hidden state.
- Strengthen the image-build semantic test to exercise the `residual=None`
  contraction path, the ordinary residual-stream path, and capture-layer
  selection.
- Refresh FlashInfer 0.6.18 main to commit `71d31b5a…`, tree `a2577ad0…` after
  reviewing the intervening variable-window PrimsTS attention change.
- Advance cache schema to `v3` because the patched runtime and FlashInfer tree
  differ from the prior candidate.

## v0.1.0-rc.3

Mixed-precision runtime-contract correction. Superseded before qualification by
v0.1.0-rc.4.

- Distinguish the BF16 repository storage total from indexed tensor bytes in
  the source completion gate.
- Serialize an exhaustive compressed-tensors runtime ignore so ordinary BF16
  linears stay unquantized while format-selected routed `FusedMoE` layers use
  MXFP4.
- Disable shared-expert fusion at serving time so the protected BF16 shared
  expert is never appended to the MXFP4 routed-expert buffer.
- Pin the corrected producer SHA-256 and retain cache schema `v2`; compiled
  SGLang and FlashInfer code is unchanged from v0.1.0-rc.2.

## v0.1.0-rc.2

Greenfield BF16-derived quantization and corrected SM120 runtime candidate.
Superseded before qualification by v0.1.0-rc.3.

- Replace the deleted FP8-derived MXFP4 artifact with a direct quantization of
  `zai-org/GLM-5.3-Flash-BF16@f12e0fe1…`.
- Quantize exactly 37,152 routed-expert projections across layers 3-45. Preserve
  every other tensor as BF16, including all 347 vision tensors and the MTP
  support projections.
- Add a fail-closed streaming quantizer that validates the complete input and
  output namespaces, protected bytes, quantization config, tensor shapes,
  round-trip probes, artifact size, package provenance, and file hashes before
  an atomic final rename.
- Pin the exact current compressed-tensors and audited llm-compressor trees used
  for the production run. A later llm-compressor main commit touched only its
  pruning scheduler and was not substituted into the already-audited run.
- Refresh FlashInfer main to commit `cbcbce48…`, tree `d3a639d6…`, and rebuild
  it for `12.0f`.
- Apply a byte-gated SM120 MXFP4 patch. The vendor loader assumed GPT-OSS's
  pairwise gate/up layout and SwiGLU-OAI constants; GLM uses contiguous halves
  and standard clamped SwiGLU `(1, 0, 10)`.
- Apply upstream PR #36708's GLM DFlash2 hidden-state capture patch and test its
  mHC contraction and capture-layer contract at image build time.
- Replace the static 3/1/4 NEXTN launcher profile with native adaptive MTP
  5/1/6, plus verifier-only and pinned DFlash2 A/B modes.
- Keep vision explicitly enabled. DFlash2 uses FA4, a 2,048-token draft window,
  and FP8 draft KV to reduce its physical pool cost.
- Stop forcibly disabling SGLang custom all-reduce. It now self-tests two-GPU
  PCIe P2P and falls back to NCCL; the selected path will be benchmarked.
- Change the default configured context ceiling to 524,288 and cache schema to
  `v2`. Actual pooled capacity remains a hardware measurement.

## v0.1.0-rc.1

Initial FP8-derived candidate. Superseded and never qualified.

- Pin `lmsysorg/sglang:glm-5.3-flash-amd64` by immutable digest
  (`sha256:44cb2eed…`, linux/amd64 manifest `sha256:0836f016…`, pushed
  2026-08-27T05:22:01Z, CUDA 13.0.3, FlashInfer 0.6.17).
- **Provenance is weaker than the sibling Qwen3.8-Flash-Next build, and this is
  recorded rather than papered over.** `glm5_next` is not in `sgl-project/sglang`
  main as of 2026-08-27, and the vendor image is built from `ADD sglang.tar.gz`
  with `SGLANG_BUILD_COMMIT=unknown`. There is no upstream commit or tree to
  verify, so the base is pinned by digest only. `verify-patches.sh` asserts what
  is actually verifiable and refuses to imply more.
- Rebuild FlashInfer 0.6.18 from source at
  `e4b7fa4b7c3ba5e17286d9c59f2bcf2ca07e0a6d` with
  `FLASHINFER_CUDA_ARCH_LIST=12.0f` and `BUILD_NVEP=0`, plus the matching cubin
  package from the same tree. The vendor wheel carries no 12.0f cubins, and
  workstation Blackwell lacks TMEM/`tcgen05`/`wgmma` so sm_100 and Hopper
  kernels do not run on it. This head is the one already qualified on these
  cards by the sibling Qwen build.
- Add build-time gates that fail before the ~40-minute FlashInfer compile:
  base CUDA version match, `glm5_next` model class importable, and the
  `mxfp4-pack-quantized` contract (GROUP strategy, `group_size` 32, uint8 E8M0
  scales, `("weight_packed", "weight_scale")` parameter names).
- Targeted a locally produced MXFP4A16 artifact quantized from
  `zai-org/GLM-5.3-Flash@04c4e9e9…`: 37,152 quantized tensors (routed experts
  only, layers 3-44 plus the layer-45 MTP block), 4.25 bits/weight, everything
  else BF16.

### Why rc.1 was rejected

- It introduced avoidable FP8-to-MXFP4 compound quantization despite the
  official BF16 source being available.
- It did not patch the vendor SM120 loader's GPT-OSS-only gate/up and activation
  assumptions, which would change GLM expert semantics.
- Its static speculative profile, context defaults, and NVFP4/MXFP4 quality
  claims had not been justified on the target hardware.
- The FP8 source, scratch output, and model-volume copy were deleted before the
  greenfield requantization began.
