# Changelog

## v0.1.0-rc.20

- Advance SGLang to `ccbda6bf675dc99a0cf2044532db0335367ded2a` / tree
  `9e123e05f9a7dba91499899a25709b97a78030e5` so GLM's native draft retains
  `modelopt_fp4` when its layer-45 routed experts are serialized FP4.
- Keep the inherited DeepSeek ModelOpt draft default unquantized and keep GLM
  draft layers in BF16 when the checkpoint explicitly ignores the complete
  layer. Build-time controls cover all three decisions.
- Add the missing SM120 capability guard for GLM's mHC pre-norm path so
  consumer Blackwell uses the supported TileLang fallback instead of calling
  unavailable DeepGEMM code on its first forward pass.
- Do not lower the absent RoPE-tail contribution as a zero-K FP8 TileLang GEMM
  for GLM's 512-wide NoPE DSA layout; SM120 rejects that otherwise empty MMA
  during layout inference.
- Retain v0.1.0-rc.19's exact FlashInfer and ModelOpt trees and bump the cache
  schema to `v12`. This is a runtime-loader correction, not a quality or
  performance claim.

## v0.1.0-rc.19

SM120 W4A16 TC-decode tile replay correction. This candidate is not yet built
or qualified.

- Advance FlashInfer to `81113d3c659f9ce692aef6cfa3ca48452d0f1e9d` / tree
  `bb8a2d438976b0b78abb8fce0f0e4a968ed8b3c6` after the Qwen E4M3-K32 control
  exposed an upstream contradiction: the auto path selected its constrained
  `K=32/N=512` FC2 tile, then custom-op replay rejected the same tile through
  the generic `K>=64` validator.
- Route both auto-selection and replay through one exact predicate, retaining
  the general tile floor and permitting `K=32` only for the TC-decode
  `N=512` FC2 specialization. Add a regression using the failing
  `N/K=2048/256`, E4M3-K32 ModelOpt shape.
- Retain the exact v0.1.0-rc.18 SGLang and ModelOpt trees and bump the cache
  schema to `v11`; no model weights or quantization calculations changed.

## v0.1.0-rc.18

Immutable-build packaging correction. This candidate is not yet built or
qualified.

- Fetch and verify ModelOpt's exact `0.47.0rc0` release tag at the already
  pinned commit before installing it. The rc.17 depth-one commit fetch omitted
  the tag, causing `setuptools-scm` to report `0.0.1.dev1+...` and correctly
  fail the locked-version assertion after SGLang and FlashInfer had built.
- Retain the exact rc.17 SGLang, FlashInfer, and ModelOpt source trees and the
  `v10` cache schema; this correction does not change runtime code or carry a
  new serving workaround.

## v0.1.0-rc.17

Greenfield E4M3-K32 W4A16 integration image. This candidate is not qualified.

- Remove the complete rc.16-and-earlier local runtime patch set and rejected
  MXFP4 serving contract from the active image.
- Shadow the unverifiable vendor SGLang tarball with exact integration commit
  `42a56dc505f775d6f54e9d27a9b57c66023420a0` and tree
  `16eb1fe669e54253b16d206a21e79e9cc7ea6132`.
- Install exact FlashInfer commit
  `008122fa75c7a27c839feea57a6ef8e8846fa265`, including the upstream large
  W4A16 bank-addressing fix and the matching E4M3-K32 SM120 preparation path.
- Install exact ModelOpt commit
  `022767c7ab3d7d36211affd85e5c496770cde768` for controlled quantization.
- Restore the desired launcher contract: TP2/EP1, vision, adaptive native MTP,
  raw-layout FP8 TileLang DSA, FP8 KV, C4/20 recurrent slots, and a 524,288
  total-token target. These are unmeasured qualification goals.
- Bump compiled-cache schema to `v10`.

## v0.1.0-rc.16

- Register post-load MLA `w_kc` and `w_vc` tensors as non-persistent buffers
  and include non-persistent buffers in SGLang's CPU-offload functional state.
  This preserves checkpoint serialization while ensuring derived execution
  tensors follow an offloaded GLM decoder layer back to its CUDA device.
- Add a build-time contract proving the buffers stay outside `state_dict`, are
  accepted by `functional_call`, and restore their original module values.
- Preserve the v0.1.0-rc.15 tied-KDA, KDA padding, SM120 TileLang, target,
  native-MTP, multimodal, and parser fixes unchanged.

## v0.1.0-rc.15

- Allow SGLang's CPU offloader to materialize GLM-5.3's tied KDA parameter
  aliases. The exact LibertAIDAI NVFP4 control otherwise loads all 121 shards
  and then fails its first warmup in `torch.func.functional_call`.
- Carry SGLang PR #36885's KDA padding correction: preserve `-1` through the
  chunked state kernel instead of remapping padding onto the last allocatable
  Mamba slot, which can silently corrupt a live request under padded batches.
- Preserve the v0.1.0-rc.14 SM120 TileLang launch and all earlier target,
  native-MTP, multimodal, and parser fixes unchanged.

## v0.1.0-rc.14

SM120 TileLang shared-memory correctness-control candidate. This candidate is
not qualified.

- Add a byte-gated launch for the independent TileLang BF16-KV no-RoPE DSA
  path. SM120/SM121 uses `block_I=32`, one pipeline stage, and 128 threads;
  non-SM12 CUDA and HIP paths remain unchanged.
- Pin the exact vendor preimage and postimage and run a semantic build-time
  contract for both SM12 and non-SM12 dispatch.
- Retain cache schema `v9`, the existing FP8 sparse-MLA path, vision support,
  and native MTP profile. The KV writer and persisted cache ABI are unchanged.

## v0.1.0-rc.13

No-RoPE MLA reserved-slot correctness candidate. This candidate is not
qualified.

- Make GLM's dedicated no-RoPE MLA scatter apply the same
  `reserved_skip_index` and DCP ownership mapping as the ordinary MLA writer.
- Add a source contract and exact-GPU regression covering both the protected
  padding slot and an ordinary positive-slot write.
- Advance the compiled-cache schema to `v9` because the effective KV writer
  changed.
- Retain the previous candidate's unfused KPool transform, pinned FlashInfer
  source, vision configuration, and native MTP profile.

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
- Use the exact-hardware capacity envelope established after v0.1.0-rc.11 was
  built: CPU image preprocessing, fixed five-step native MTP, concurrency one,
  batch-one CUDA graphs, five recurrent-state slots, and PyTorch's default
  allocator. This retains vision and native MTP in the capacity-first profile.

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

## v0.1.0-rc.10

Exact GLM-5.3 no-RoPE sparse-MLA attempt. No image was published: the build
failed closed when the final adapter postimage did not match the exact vendor
base. The failure is preserved in
[`evidence/v0.1.0-rc.10-build-failure-20260828.txt`](evidence/v0.1.0-rc.10-build-failure-20260828.txt).

- Correct the cache-layout diagnosis: GLM-5.3 uses SGLang's generic 528-byte
  cache (512 E4M3 latent bytes plus four arbitrary FP32 scales), not the
  584-byte DSv4 footer cache assumed by v0.1.0-rc.9.
- Add an isolated FlashInfer SM120 model trait for the 512-dimensional no-RoPE
  query/cache geometry while leaving existing cache layouts unchanged.
- Preserve 2,051 KPool entries by padding with `-1` to one 2,176-wide TP=2
  dispatch for both decode and prefill.
- Compile the entire sparse-MLA module during the image build and retain
  exact-SM120 numerical decode/prefill tests for qualification.
- Advance cache schema to `v7`.

## v0.1.0-rc.9

GLM-5.3 DSv4/KPool sparse-MLA correctness candidate. Not qualified.

- Route the model's 584-byte footer cache and complete KPool-extended DSA table
  through FlashInfer's supported 128+1,923 dual-segment SM120 ABI.
- Point both segments at the same physical packed cache and use full segment
  lengths so base-table padding cannot hide live KPool tail entries.
- Preserve the unrelated 656-byte GLM-NSA adapter and fail closed for every
  other KPool/backend combination.
- Add build-time adapter/guard contracts and archive the exact-hardware
  numerical regression used before this immutable build.
- Refresh FlashInfer main to commit `950376c4…`, tree `d44c17d4…`.
- Advance compiled-cache schema to `v6`.

## v0.1.0-rc.8

Source-JIT and MXFP4 post-loader correctness candidate. Not qualified.

- Correct the SM120 MXFP4 post-loader to read `moe_runner_config` from the
  `Mxfp4MoEMethod` that owns it, and make that ownership part of the build-time
  contract.
- Remove the generic cross-architecture `flashinfer-cubin` package whose
  all-or-nothing network build prevented v0.1.0-rc.6 from publishing.
- Retain the exact FlashInfer source pin, force offline source JIT at runtime,
  and assert that neither optional artifact package is installed.
- Include v0.1.0-rc.7's GLM47 parser backport.
- Retain compiled-cache schema `v5`; no kernel source changes after
  v0.1.0-rc.6.

## v0.1.0-rc.7

GLM47 streaming tool-parser correctness candidate. Superseded before build by
v0.1.0-rc.8.

- Backport the GLM47-specific part of merged SGLang #36626.
- Close the outer streaming JSON object after a nested object argument and
  resolve tool properties beneath legal top-level composite schemas.
- Fail closed on both vendor Python-file preimages and run flat,
  nested-object, and composite-schema contracts during the image build.

## v0.1.0-rc.6

Native GLM-5.3 SM120 sparse-MLA compatibility candidate. Not qualified.

- Keep FP8 E4M3 KV while routing both DSA phases through upstream #26928's
  dedicated FlashInfer sparse-MLA kernel for SM120/SM121. Generic TRT-LLM MLA
  aborts on workstation Blackwell with `Unsupported architecture`.
- Extend only the fail-closed GLM architecture allowlist to
  `Glm5NextForConditionalGeneration` and its NextN class. The vendor guard
  otherwise rejects GLM-5.3 after both target and MTP weights load.
- Assert exact vendor preimage/postimage hashes and run positive and negative
  architecture, device, KV-dtype, and paired-backend contract checks.
- Archive exact draft PR #36745 head `f14393b2…` after the vendor preimage
  failed all four upstream exact-tie, order, overflow, and concurrent-stream
  GPU regressions on the target SM120 card.
- Advance compiled-cache schema to `v5`.

## v0.1.0-rc.5

SM120 mHC/indexer fallback candidate. Superseded before qualification after the
exact-hardware diagnostic exposed the stale sparse-MLA architecture allowlist.

- Bake the same seven SM120 fallback settings SGLang currently applies to
  `DeepseekV4ForCausalLM`; GLM's shared DeepSeek/DSA hook does not inherit that
  architecture-specific block.
- Disable the unavailable DeepGEMM TF32 mHC prenorm path. An exact-hardware
  v0.1.0-rc.3 diagnostic loaded the target and native MTP weights, then
  reproduced upstream issue #29738's `deep_gemm` `NameError` during warmup.
- Select the SM120-safe top-k, mHC, FP8 paged-MQA, and TileLang indexer settings
  as one coherent runtime contract.
- Record the hash-verified 184,945,092,190-byte production artifact and advance
  compiled-cache schema to `v4`.

## v0.1.0-rc.4

DFlash mHC correctness and refreshed FlashInfer candidate. Superseded before
qualification after the exact-hardware diagnostic exposed the missing GLM
SM120 runtime guard.

- Apply upstream PR #36755 after #36708 so DFlash hidden-state capture accepts
  the real mHC `residual=None` contract.
- Strengthen the build-time test to exercise `residual=None`, the ordinary
  residual stream, and capture-layer selection.
- Refresh FlashInfer 0.6.18 main to commit `71d31b5a…`, tree `a2577ad0…`.
- Advance compiled-cache schema to `v3`.

## v0.1.0-rc.3

Mixed-precision runtime-contract correction. Superseded before qualification by
v0.1.0-rc.4.

- Pin the final BF16 quant producer with separate storage/tensor byte gates and
  an exhaustive runtime ignore for ordinary BF16 linears.
- Disable shared-expert fusion so the protected BF16 shared expert is not
  appended to the MXFP4 routed-expert buffer.
- Keep cache schema `v2`; compiled SGLang and FlashInfer code is unchanged.

## v0.1.0-rc.2

BF16-derived quant and corrected SM120 runtime candidate. Superseded before
qualification by v0.1.0-rc.3.

- Point the runtime contract at the immutable official BF16 source revision and
  the new routed-expert-only MXFP4 artifact path.
- Refresh FlashInfer main and its exact tree pin.
- Add byte-gated GLM SM120 MXFP4 layout/activation semantics and DFlash2
  hidden-state capture patches, with build-time semantic tests.
- Add verifier-only, adaptive native MTP, and DFlash2 modes while keeping vision
  explicitly enabled.
- Move compiled-cache schema to `v2`.

## v0.1.0-rc.1

Initial candidate. Not yet built, not yet qualified on hardware.

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
- Target the locally produced MXFP4A16 artifact quantized from
  `zai-org/GLM-5.3-Flash@04c4e9e9…`: 37,152 quantized tensors (routed experts
  only, layers 3-44 plus the layer-45 MTP block), 4.25 bits/weight, everything
  else BF16.

### Known limitations at rc.1

- **No native SM120 MXFP4 MoE kernel.** FlashInfer #2847 and vLLM #31085 leave
  three runtime guards filtering SM120 (SM103/TRTLLM modules filter to
  `supported_major_versions=[10]`; SM120 is major 12). Expect the Marlin
  fallback — measured ~28 % slower prefill on gpt-oss-120b. Carrying that patch
  is the intended rc.2 work.
- sglang #36596, #36653, #36599, #36550, #36669 are open and relevant; see
  `stack.lock.json` → `known_limitations`.
