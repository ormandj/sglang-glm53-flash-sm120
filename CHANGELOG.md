# Changelog

## v0.1.0-rc.10

Exact GLM-5.3 no-RoPE sparse-MLA candidate. This candidate is not qualified.

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
