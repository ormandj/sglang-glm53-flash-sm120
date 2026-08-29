# Upstream preparation dossier

This file prepares future upstream work without submitting pull requests. Each
item must be rebased, minimized, independently tested, and supported by a
reproducer before submission. GLM-specific integration code may request a
generic capability; low-level kernels should not contain a model-name pin.

## Submission principles

- Prefer one independently reviewable failure/fix per pull request.
- Keep generic paths and non-SM120 behavior byte-for-byte unchanged when the
  capability is absent.
- Key static tactics by architecture, dtype, exact shape, and semantic flags,
  not by model name.
- Retain autotuning for unrecognized shapes and devices.
- Distinguish checkpoint/loader contracts from kernel correctness and memory
  ownership fixes.
- Include CPU tests for selection and ownership logic plus exact-GPU tests for
  launch, numerical agreement, memory behavior, and performance.
- Do not claim provenance for the vendor `glm5_next` source that cannot be
  verified against upstream history.

## Proposed SGLang pull-request splits

### 1. ModelOpt FP4 NextN checkpoint contract

Problem: inherited DeepSeek NextN logic suppresses ModelOpt FP4 for drafts, but
GLM layer 45 can contain serialized W4A16 routed experts. Conversely, a
checkpoint that explicitly ignores the complete draft layer must remain BF16.

Upstream-shaped fix: derive the draft quant decision from the checkpoint's
`quantization_config.ignore` contract and the actual draft layer index. Keep
DeepSeek's existing default. Add tests for target quantized/draft quantized,
target quantized/draft explicitly ignored, and inherited DeepSeek behavior.

### 2. In-place W4A16 prepared-weight ownership

Problem: functional preparation retained a second process-lifetime expert bank
and exhausted HBM after a successful model load.

Upstream-shaped fix: prepare byte-compatible weights once during layer/model
loading, reuse input storage where layout permits it, retain the prepared views
on the layer quant-info object, and dispatch only through those views. Fail
closed when a runner requiring prepared weights receives none.

Evidence needed: source/prepared storage alias assertions, numerical kernel
agreement, target-only before/after HBM trace, and a regression showing no
process-lifetime cache growth across layers or requests.

### 3. Generic pre-pool kernel compilation hook

Problem: large-prefill Triton/CuTe specializations can first compile after KV,
recurrent-state pools, and CUDA graphs have consumed nearly all HBM. Candidate
cubins and compiler workspaces then OOM a healthy server.

Upstream-shaped fix:

- expose a model hook after weights are loaded but before memory pools are
  committed;
- provide generic helpers that accept device, dtype, tensor geometry, and
  semantic flags;
- use a device/shape registry for measured static winners;
- compile both short-prefill and configured chunk-prefill shape buckets;
- leave ordinary autotuning untouched for shapes absent from the registry.

The first measured SM120 KDA entries are BF16 H=32, K=128, V=128, chunk 64,
subchunk 16, varlen, with separate fused-small-grid and unfused-large-grid
semantics. The exact tactics must be preserved in an evidence file, not encoded
as a GLM name check.

Evidence needed: CPU selector tests; exact-GPU numerical equality; cold-start
compile trace proving compilation precedes pool allocation; no compiler-caused
HBM growth on the first 8,192-token request; warm-restart cache reuse; and
performance within noise of the autotuned winner. Backend request workspace
must be measured separately rather than attributed to compilation.

### 4. SM120 TileLang DSA low-shared-memory schedule

Problem: the datacenter-oriented schedule requests 169,984 bytes of dynamic
shared memory and cannot launch on RTX PRO 6000 SM120.

Upstream-shaped fix: capability/shared-memory-budget selection of the measured
one-stage schedule. Avoid a blanket SM120 override when current dispatch
already selects a safe launch.

Evidence needed: failing launch receipt, successful exact numerical comparison,
reported opt-in shared-memory limit, and non-SM12 selection regression.

### 5. Zero-tail FP8 TileLang dispatch guard

Problem: GLM's no-RoPE geometry can present an absent RoPE-tail term as a zero-K
FP8 GEMM. Lowering that empty contribution through the ordinary MMA path fails
during layout inference.

Upstream-shaped fix: skip only the mathematically empty contribution. Test
zero-tail and nonzero-tail paths independently and compare the surviving sum to
a reference.

### 6. SM120 mHC pre-norm capability guard

Problem: a DeepGEMM mHC pre-norm path can be selected on SM120 even when the
required implementation is unavailable, failing on the first forward pass.

Upstream-shaped fix: make backend availability part of selection rather than
assuming all Blackwell devices share datacenter support. Preserve current
behavior where the capability exists.

### 7. DSA index-prefix gather precompile helper

Problem: the 132-byte FP8 index-row gather can JIT on the first large prefill
after pools are allocated.

Upstream-shaped fix: a generic helper keyed by device, page size, index head
dimension, and dtypes. The GLM model hook should call it with its actual shape;
the helper must not mention GLM.

### 8. W4A16 route-pack precompile helper

Problem: the large-prefill route block selection produces CuTe route-pack
specializations that can compile after readiness.

Upstream-shaped fix: compile the selected route block from the configured
prefill-token count, top-k, and expert count before pools. Do not pin only
288 experts or top-k 8 in the API; unsupported combinations may return false
or use the normal path.

### 9. Native SM120 no-RoPE sparse-MLA integration

Problem: GLM's DSA geometry uses a 512-dimensional no-RoPE latent, 2,051 logical
candidate entries, and scaled FP8 cache metadata that differ from existing
datacenter-oriented traits.

Upstream-shaped fix: explicit architecture/configuration admission, padding the
candidate table to the physical kernel contract without mapping sentinels to a
live slot, one persistent runner, and scratch carved from an existing workspace
where safe. Backend validation must apply only when the native backend is
selected so TileLang remains an independent control.

The admission/padding regression must use the real two-stage geometry:
GLM-5.3 declares `qk_nope_head_dim=256`, while absorption produces the
512-wide query consumed by sparse MLA. A guard on the configured dimension
silently skips padding and crashes the scheduler on the first 2,051-wide large
prefill. Detect from the tensor/cache contract or pass an explicit admitted
model-family enum; do not duplicate the wrong 512 value in the test fixture.

This work depends on the corresponding FlashInfer API and should be proposed
only after corrected cache-layout and ownership measurements.

### 10. Recurrent-state sentinel correctness

Problem: remapping `-1` padding to the last allocatable KDA/Mamba slot can
silently modify a live request under padded batches.

Upstream-shaped fix: preserve the sentinel through kernels that explicitly
support padding or allocate a dedicated non-live sink. Do not advertise this as
a capacity optimization. Add a concurrent-request corruption regression.

### 11. Share NextN embedding/head before pool sizing

Problem: the draft correctly aliases the target embedding and output head, but
only after the target pool has already been sized and allocated. On GLM-5.3 at
TP2 the temporary placeholders are 1.181640625 GiB per rank. They are not
resident duplicates, but they inflate startup peak and hide memory from
automatic pool sizing.

Upstream-shaped fix: perform the existing delete/rebind operation immediately
after the draft model loads, before target pool configuration. Make the helper
idempotent because legacy allocation flows may still invoke it. Evidence must
separate lower peak/corrected sizing from steady-state memory, which is already
shared today.

### 12. Workspace-aware TileLang FP8 DSA long prefill

Problem: the partial-plus-combine path owns request-sized FP8 query, partial
output/LSE, and BF16 combine-output buffers in addition to live model
activations. At an 8,192-token chunk on TP2, the combine output alone is
256 MiB; the exact v0.1.0-rc.25 failure occurred at this allocation after the
process reached 94.81 GiB used per rank.

Upstream-shaped fix: expose an exact workspace estimator and select a direct,
more heavily fused, or smaller-chunk path from available HBM. A configuration
that cannot reserve its request workspace should fail during startup instead of
crashing a healthy scheduler. Compare alternative inner-iteration grouping for
large query batches, where sequence parallelism may already saturate the GPU.

## Proposed FlashInfer pull-request splits

### 1. SM120 no-RoPE sparse-MLA trait and cache contract

Define the 512-dimensional no-RoPE query/cache geometry, FP8 scales, physical
top-k width, and public wrapper API without a GLM name dependency where the
shape contract is sufficient. The native packed row is 528 bytes: 512 FP8
latent bytes plus four inline FP32 scales. A wider runtime row stride may be
accepted for shared cache groups, but it must not force persistent 128-byte
padding when no RoPE payload exists. SGLang's separate 132-byte index row is
additional storage; document this explicitly because ambiguity directly
changes shared-token capacity.

### 2. Persistent scratch/workspace ownership

Make scratch lifetime explicit. Prefer caller-owned or persistent workspace
views over per-request allocations. Prove that repeated long prefills and C4
decode do not grow HBM after warmup.

### 3. W4A16 preparation/storage reuse

Where not already upstream, accept prepared ModelOpt E4M3-K32 views whose
backing allocations are the original serialized tensors. Ensure route replay,
tile predicates, and custom-op validation accept exactly the same shapes as
selection.

### 4. Large-prefill route-pack ahead-of-time preparation

Expose a narrow public preparation API for the route-pack specializations
selected from token count, top-k, and expert count. This should support the
SGLang pre-pool hook without importing private implementation modules.

## Potential ModelOpt work

The local producer uses ModelOpt's latest static-MSE sweep and joint gate/up
calibration. Before proposing changes, determine whether an official exported
API can produce the same flat W4A16 checkpoint without depending on private
helpers. A useful upstream contribution would be an audited MoE W4A16 export
recipe with:

- joint gate/up global-scale invariants;
- K=16 and K=32 selection;
- source-exact ignore lists for vision, attention, shared experts, and draft
  support modules;
- atomic publish and round-trip tensor validation;
- reconstruction metrics and producer metadata.

No quality claim should be based on reconstruction alone.

## Work that should not be submitted

- model-name checks in Triton tactic selection;
- a global reduction of TileLang shared memory when the current dispatcher is
  already safe;
- mapping padding sentinels onto a live recurrent slot;
- disabling vision, native MTP, or FP8 KV to hide memory failures;
- reducing model precision before exhausting storage ownership, duplicate
  allocation, workspace, graph, and cache-layout fixes;
- treating the rejected local MXFP4 artifact's garbled output as evidence of a
  current SGLang kernel defect;
- pinning an exact tactic without numerical, memory, and performance evidence
  on the keyed hardware/shape.

## Pre-submission checklist

For every eventual pull request:

1. rebase onto current upstream main and search for superseding work;
2. reduce to one causal fix and remove historical experiments;
3. add a minimal reproducer and exact failure signature;
4. add CPU selection/contract tests where possible;
5. add exact-SM120 numerical and memory evidence;
6. compare against the unmodified upstream path and relevant open pull
   requests;
7. document performance only on the exact tested commit and image;
8. verify licenses and avoid copying unverifiable vendor provenance;
9. prepare commit messages and pull-request text locally;
10. stop before submission unless the user explicitly authorizes it.
