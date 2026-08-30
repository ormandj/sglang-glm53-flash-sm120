# Changelog

## v0.1.0-rc.53 (pre-top-k logits ownership candidate, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `aae81abce55f5c894a6a072fc7f8b6853935b55e` and the reproducible patched tree
  to `91dbea66838d48d4c80c2cb25ee834d0199ee90a`.
- Records the v0.1.0-rc.52 result: its complete focused GPU suite passed, but
  the exact C4 serving profile failed during cold wave 1. The radix overwrite
  covered 26,880 int64 slots whose halves decode as 53,760 plausible FP32
  scores, exactly the target-verify paged-logits shape `24 x 2240`.
- Moves the disabled paged-logits owner handoff before fused top-k and returns
  the real DeepGEMM view through that boundary. The preceding post-consumer
  handoff did not prevent the exact-shape overwrite even though its synthetic
  ownership test passed. No DSA kernel, tensor value, graph shape,
  quantization, vision, MTP, or KV format changes.
- Adds an exact-shape CUDA regression that executes the real DeepGEMM paged-MQA
  producer under the served outer compile mode, verifies capture ownership,
  and confirms another CUDA graph sharing the pool cannot reuse or overwrite
  the owned allocation. A read-only Claude review rejected broad method-level
  disabling and unproven top-k retention; both were removed from this scoped
  candidate.
- Uses fresh cache schema `v40`. This candidate remains unqualified pending the
  exact-image GPU gate, sustained C4 serving, and matched memory/performance.

## v0.1.0-rc.52 (paged-logits ownership candidate, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `7eff1bd551e57e83583f3546e745d0f310027af5` and the reproducible patched tree
  to `c7ba3809e6a4c976dc73b4e94eba71e04f80adf4`.
- Records the v0.1.0-rc.51 result: the complete gate ownership GPU suite passed
  and 13 productive C4 waves completed, but wave 14 returned zero tokens before
  both ranks found all 527 allocator entries overwritten by paired signed-FP32
  values. This disproves the compiled head gate as the sole remaining writer.
- Gives each full graph shape bounded ownership of `_get_topk_paged`'s FP32
  paged-MQA logits output. The tensor is consumed by captured top-k work but is
  not returned by the model; its previous Python diagnostic was also absorbed
  before the top-k boundary by the enclosing compile. Only its ownership and
  diagnostic handoff are kept outside Dynamo; paged-MQA and top-k math remain
  compiled unchanged.
- Adds CPU outer-compile and CUDA outer-compile/capture/replay regressions for
  the paged-logits owner. Model values, quantization, vision, MTP, FP8 KV, DSA
  kernels, and graph shapes are unchanged. Uses fresh cache schema `v39`; this
  candidate remains unqualified pending exact-image GPU tests, memory fit,
  sustained serving, and matched performance.

## v0.1.0-rc.51 (replay oracle correction, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `d570a59039897ced3c3b25ec8521d8dfe0ddc252` and the reproducible patched tree
  to `b60bacdca75653b1fa85952c3bb9ad2c162676fe`.
- Corrects the real CUDA graph replay oracle to compare the first replay with
  an already-executed outer-compiled reference. CUDA capture records the graph
  but does not populate the captured output before replay; v0.1.0-rc.50's
  synchronized pre-replay clone therefore still held stale storage.
- Retains the v0.1.0-rc.49 runtime fix byte-for-byte. Both v0.1.0-rc.49 and
  v0.1.0-rc.50 passed the CPU outer-compile regression and the CUDA direct and
  outer owner assertions before their replay-oracle failures. Model values,
  quantization, vision, MTP, FP8 KV, DSA kernels, and graph shapes are
  unchanged. Uses fresh cache schema `v38`; this candidate remains unqualified
  pending the complete GPU gate, sustained serving, and matched performance.

## v0.1.0-rc.50 (test synchronization, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `d502134bc84e319198078c84b9f313ed42eab5de` and the reproducible patched tree
  to `622f994b2cde2c2bf2ca82d1611a51b15cfd3bbc`.
- Synchronizes the CUDA replay test's asynchronous expected-value clone before
  replay begins. v0.1.0-rc.49 passed all five CPU ownership tests and reached
  the CUDA replay comparison, but the test allowed replay to race that clone;
  this was a test-oracle defect, not a runtime ownership failure.
- Retains the v0.1.0-rc.49 runtime fix byte-for-byte: only the Python owner
  handoff is outside Dynamo, while gate math, model values, quantization,
  vision, MTP, FP8 KV, DSA kernels, and graph shapes remain unchanged. Uses
  fresh cache schema `v37`; this candidate remains unqualified pending the
  complete exact-image GPU gate, sustained serving, and matched performance.

## v0.1.0-rc.49 (outer-compile ownership candidate, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `de5b13df5da97a70b290157b44e03296ea9e74c5` and the reproducible patched tree
  to `8d99bf2e8b5eab0a0653c2015e30e979bd9e068c`.
- Records the exact v0.1.0-rc.48 failure: nine C4 waves completed before wave
  10 overwrote all 527 live allocator entries. A separate exact-image test
  reproduced the implementation gap directly: the gate wrapper retained one
  owner when called directly and zero through SGLang's enclosing
  `torch.compile` path.
- Keeps the DSA gate math compiled while placing only the Python capture-owner
  handoff behind `torch.compiler.disable`, so Dynamo cannot fold away the
  active capture scope. Removes the prior missing-scope hard failure to avoid a
  regression on the supported breakable backend.
- Adds a CPU outer-compile regression, CUDA outer-compile retention, and real
  CUDA capture/replay coverage. The final source shape incorporates a read-only
  Claude review. Model values, quantization, vision, MTP, FP8 KV, DSA kernels,
  and graph shapes are unchanged. Uses fresh cache schema `v36`; this candidate
  remains unqualified pending exact-image GPU tests, sustained serving, and
  matched startup/prefill/decode measurements.

## v0.1.0-rc.48 (test correction, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `c049839c0ecaf0e5b79e8d8c46addcc34b84a568` and the reproducible patched tree
  to `6c706bf37853ac96f2bb4a001fbe92bcaadf4455`.
- Corrects the focused CPU regression so its mocked compiled DSA gate remains
  active for both the capture-scoped call and the subsequent eager call. The
  v0.1.0-rc.47 exact-image Job stopped at this test before executing any CUDA
  assertion because the second call reached an intentionally uninitialized
  synthetic `IndexerKPool` after the mock had ended.
- Retains the complete v0.1.0-rc.47 runtime fix byte-for-byte: compiled DSA
  gate outputs remain capture-owned per full graph shape, and model values,
  quantization, vision, MTP, FP8 KV, optimized kernels, and graph shapes are
  unchanged. Uses fresh cache schema `v35`; this candidate remains unqualified
  pending the complete exact-image GPU gate and sustained serving.

## v0.1.0-rc.47 (correctness candidate, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `8ec12a81db64d86419c2e0d3ddfb3bd35e83c298` and the reproducible patched tree
  to `3d6c940262478a1818972312c3318c669090cd06`.
- Records the exact v0.1.0-rc.46 sustained-serving failure: after eleven warm
  C4 waves, the next captured decode overwrote all 527 live allocator entries
  with plausible paired FP32 activations. This disproves the eager-extend
  lifetime boundary tested by v0.1.0-rc.46.
- Restores the compiled DSA head-gate path for every execution mode and gives
  each full CUDA graph shape bounded ownership of the compiled output consumed
  by its captured top-k kernels. Recapture replaces the prior owner set,
  cleanup releases it, eager execution retains nothing, and unsupported CUDA
  capture without an owner scope fails explicitly.
- Adds focused ownership, replacement, cleanup, and missing-scope tests. Model
  values, quantization, vision, MTP, FP8 KV, optimized prefill/decode kernels,
  and full decode graph shapes are unchanged. Uses fresh cache schema `v34`;
  this candidate remains unqualified pending exact-image GPU tests, sustained
  serving, and matched performance measurements.

## v0.1.0-rc.46 (correctness candidate, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `df8325b1824d8f22aaa82acb11ed2ece56e4a879` and the reproducible patched tree
  to `6207dca678cdab8ead0b56877d17f035bf787f98`.
- Uses shared eager DSA head-gate math only for ordinary uncaptured extend, where
  v0.1.0-rc.45's 64-token failure carried the exact signed-FP32 size signature
  of the dynamically compiled output. Decode, target verification, draft
  extension, torch.compile, and CUDA-graph modes retain the upstream compiled
  fast path. This dispatch boundary incorporates a read-only Claude review.
- Extends the failure-only page diagnostic with invalid count and first/last
  invalid positions to identify the exact overwrite extent if the candidate
  does not resolve the allocator mutation.
- Retains model values, quantization, vision, MTP, full decode graphs, DSA
  kernels, and FP8 KV. Uses fresh cache schema `v33`; this candidate remains
  unqualified pending exact-image tests, sustained serving, and matched
  prefill/decode measurements.

## v0.1.0-rc.45 (test-profile correction, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `69bb461a90a2deee1323653bcad31dbdb42c7e75` and the reproducible patched tree
  to `f059337d81308c985593dac980878b3f3c23082a`.
- Makes the standalone fused MHC GPU regression explicitly select the TileLang
  prenorm fallback used by the GLM SM120 serving profile. The v0.1.0-rc.44
  in-image run otherwise inherited the generic DeepGEMM-on default and failed
  before its non-empty fused comparisons because that optional backend is not
  available in the image.
- Retains the exact v0.1.0-rc.44 runtime fix, model values, quantization,
  vision, MTP, full decode graphs, prefill, and DSA backends. Uses fresh cache
  schema `v32`; this candidate remains unqualified pending exact-image tests
  and sustained distinct-prefix C4 validation.

## v0.1.0-rc.44 (correctness fix, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `3477f7b5188df80ffce5be91b940bb3ff242ee7f` and the reproducible patched tree
  to `1a7fc14d8216ba3f2bfcfc88b76d11017940872b`.
- Extends capture-scoped ownership to the two local FP32 split-K scratch
  tensors in the SM120 fused MHC post/pre path. The first ownership change
  covered the standalone prenorm writer but rc.43 later reproduced the same
  512-byte overwrite signature from this separate fused writer.
- Adds focused CUDA regression coverage for empty and non-empty fused calls,
  including exact owner count, dtype, rank, and shape relationships.
- Keeps quantization, model values, vision, MTP, full decode graphs, prefill,
  and DSA backends unchanged. Uses fresh cache schema `v31`; this candidate is
  unqualified until its exact image passes in-image tests and sustained
  distinct-prefix C4 validation on the two RTX PRO 6000 Blackwell GPUs.

## v0.1.0-rc.43 (correctness fix, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `46c49ca6696fe14d9290a55d0d8738b28a255f0d` and the reproducible patched tree
  to `3b7a01864a7f87fb750191c15615676b32ce32d3`.
- Makes full CUDA graphs retain the optimized MHC prenorm split-K scratch
  tensors for the lifetime of each captured shape. Recapture replaces the
  prior owners and graph cleanup releases them, so eager execution and graph
  replay do not accumulate per-request state.
- Keeps optimized MHC prenorm, full decode graphs, DSA backends, quantization,
  vision, FP8 KV, and adaptive MTP enabled. It also adds focused ownership
  regression coverage and corrects the allocator snapshot test's environment
  mock target.
- Uses fresh cache schema `v30`. This is a correctness candidate; it remains
  unqualified until the exact image passes the in-image unit test and sustained
  distinct-prefix C4 validation on the two RTX PRO 6000 Blackwell GPUs.

## v0.1.0-rc.42 (diagnostic, not yet built or qualified)

- Advances the internal SGLang integration working head to
  `f9ec47104ecc21474ce440551e6a4a665425d45a` and the reproducible patched tree
  to `e073130636827c183bbc874b672f039d9cbaed1f`.
- Appends captured DSA graph-buffer overlap or nearest-range context to the
  first debug-gated paged-allocator range failure. This closes the diagnostic
  ordering gap observed in the rc.41 full-speed control, where the initial
  range assertion terminated the scheduler before the existing overlap report.
- Uses fresh cache schema `v29`. Model values, quantization, vision, MTP,
  serving kernels, tensor-retention policy, and synchronization policy are
  unchanged. This candidate is fault localization, not a correctness fix or
  qualification claim.

## v0.1.0-rc.41 (diagnostic, not yet built or qualified)

- Fetches exact official SGLang and FlashInfer base commits, then applies the
  two project-owned integration patches stored in this internal repository.
  The build verifies both patch checksums, both upstream trees, and both final
  integration trees before compiling or installing any source.
- Retains editable pre-validation history in the internal `homelab/sglang` and
  `homelab/flashinfer` working repositories. This avoids both personal-GitHub
  publication and cross-repository runner credentials while preserving a clean
  path to focused upstream pull requests after exact-hardware validation.
- Carries the same DSA CUDA-graph buffer lifetime probe and final integration
  trees intended by v0.1.0-rc.40. Cache schema `v28` prevents reuse of artifacts
  from the failed source-fetch candidate.
- This candidate is fault localization, not a correctness fix or qualification
  claim.

## v0.1.0-rc.40 (build failed; no image published)

- Moves the pre-validation SGLang and FlashInfer integration provenance from
  personal GitHub forks to the internal `homelab/sglang` and
  `homelab/flashinfer` Forgejo repositories. Official upstream repositories
  remain the source and eventual pull-request targets after validation; all
  candidate builds and project-owned integration branches remain internal.
- Both Forgejo jobs failed before image construction because clean runners
  cannot anonymously clone the private working repositories. No repository
  visibility or credential policy was changed; v0.1.0-rc.41 instead carries
  the integration deltas inside the container repository.
- Advances the integration fork to head
  `0d691b6ddf0d3e27ebb7343e78a64b203eb6949c` and tree
  `179d7d1440c7f0ea800457718966097d4af33140`.
- Adds a bounded, opt-in DSA CUDA-graph buffer lifetime probe for the paged
  logits and fused top-k outputs. Probe mode records logical and backing-storage
  address ranges without retaining tensors; retain mode additionally preserves
  the captured owners for a controlled lifetime comparison.
- Allocator and radix diagnostics report exact overlap or nearest adjacency to
  those captured ranges when corruption is detected. The serving kernels,
  quantization, vision path, and default runtime behavior are unchanged while
  the probe is disabled.
- This candidate is fault localization, not a correctness fix or qualification
  claim.

## v0.1.0-rc.39 (diagnostic, not yet built or qualified)

- Corrects the diagnostic module's environment import and advances the signed
  SGLang integration to head
  `66e03150e9611a61293bd8cc9fd640664dbe6c10` and tree
  `216200b6a28d440886a66a5b0841fded40fed652`.
- The corrected module imports successfully with the exact SGLang interpreter
  in the running rc.37 container. Local syntax, formatting, import order, lint,
  test registration, and whitespace checks also pass.
- rc.38 passed repository validation but failed during the immutable build on
  the incorrect import, before any rc.38 image was published.
- The allocator snapshot diagnostic and runtime envelope are otherwise
  unchanged. This remains fault localization, not a correctness fix or
  qualification claim.

## v0.1.0-rc.38 (diagnostic, not yet built or qualified)

- Advances the SGLang integration to signed head
  `f00e62495e3bc3a7506d53e16087d91153e98de0` and tree
  `cbd0e6f99180ea31e46d12b14cfdbbe5200b53b3`.
- Snapshots the paged allocator state after every mutation boundary and compares
  it before the next operation, distinguishing an invalid release from a later
  write into state that was valid when the prior operation completed.
- Makes the existing diagnostic checks at both paged release entry points
  synchronous so the first invalid input is attributed at its source.
- Adds a focused CPU regression for unchanged state and for attribution of a
  change between allocator operations. The local checkout has no PyTorch, so
  execution is delegated to the immutable image build; syntax, formatting,
  import order, lint, test registration, and whitespace checks passed locally.
- Keeps the rc.37 runtime envelope unchanged. This remains fault localization,
  not a correctness fix or qualification claim.

## v0.1.0-rc.37

- Pin SGLang integration commit
  `258f815b78935fc46d678e48b1e76af6296c4e1b` / tree
  `28244e492856486856a8f001190661b3592a68ff` on the unchanged
  `cdbfe90b4a6c728e03e6520862d792501b3a97bb` upstream-main base.
- The exact v0.1.0-rc.36 diagnostic caught the first bad whole-tree state when
  finished-request insertion completed, after the fresh input and the
  new-node, unevict, and split boundaries remained clean. Add debug-gated,
  per-node Full-value storage snapshots around every insert action, reject
  frees that alias reachable Full storage, and report the exact node, parent,
  key length, pointer, offset, checksum, and changed values at the first
  boundary. Advance the compiled-cache schema to `v24`. This is an unbuilt,
  unqualified localization candidate, not a claimed crash fix or performance
  result.

## v0.1.0-rc.36

- Rebase the exact SGLang integration onto upstream main
  `cdbfe90b4a6c728e03e6520862d792501b3a97bb` and pin integration commit
  `4b30b052e089ca18f6abd0c6052d57d953d6fa89` / tree
  `6f91718151274b53825c0da1e51a1e496d3de791`.
- The exact rc.35 repeated distinct-suffix C4 diagnostic proved that the
  invalid full-KV radix value detected at eviction is an upper-bound violation
  of at least 65,600, not a negative sentinel or reserved slot zero. Add
  diagnostic-gated synchronous value attribution at insert, new-node,
  unevict, split, whole-tree insert and match, and eviction boundaries so the
  next exact reproduction identifies where that value first appears. Advance
  the compiled-cache schema to `v23`. This is an unbuilt, unqualified
  localization candidate, not a claimed crash fix or performance result.

## v0.1.0-rc.35

- Pin SGLang integration commit
  `26a382e1d2c07ce5f99a317b6b8572f9814f6fe5` / tree
  `56005c6d183c04e3869774c11598c52639322463` on unchanged upstream-main base
  `4d53767b09429c67a4137352c762372923853eb6`.
- Extend the debug-gated lifecycle diagnostic to full-KV radix insertion, node
  split, and eviction. Distinguish an unmasked negative sentinel from reserved
  physical slot zero and emit only numeric finish-length metadata while the
  diagnostic gate is active. Advance the compiled-cache schema to `v22`.
  This is an unbuilt, unqualified localization candidate, not a claimed crash
  fix or performance result.

## v0.1.0-rc.34

- Rebase the exact GLM integration onto upstream main
  `4d53767b09429c67a4137352c762372923853eb6` and pin integration commit
  `ba6dae453ec4bc829f1830ae5429e6f5386f7480` / tree
  `8e11a5e0c3dccf79b5d0a615f8405bf1d826797c`.
- Add debug-gated paged-allocator boundary probes after the rc.33 repeated-wave
  diagnostic located a negative target MLA write location. Attribute the prefix
  tail, free-page list, extend output, and free inputs independently without
  changing the ordinary serving path. Advance the compiled-cache schema to
  `v21`. This is an unbuilt, unqualified localization candidate, not a claimed
  crash fix or performance result.

## v0.1.0-rc.33

- Rebase the exact GLM integration onto upstream main
  `46ccd7ce3e70455a971e6a7f7765cd78bc246322`, then carry the focused open
  upstream fixes sglang#36661 and sglang#36696. The former ties overlap-batch
  device-tensor lifetime to result completion instead of a fixed two-launch
  ring; the latter registers Mamba radix split nodes under their own key and
  enables a debug-gated child-key invariant for the page-size-64 DSA profile.
- Pin integration commit `1db7b4c2f24c1768ee796759441ab391005e1e3b` /
  tree `c298bcc12e657cf6a65239a86eb32b615dc0eb1b`. Retain the private-stream
  cuBLAS8 workspace profile and advance the cache schema to `v20`. This is an
  unbuilt, unqualified repeated-wave A/B; the two fixes remain separately
  attributable integration commits.

## v0.1.0-rc.32

- Rebase the exact SGLang integration tree onto upstream main
  `0a585d5bb108cab8f0922b483d7f55812f05e245`, producing integration commit
  `33121e7a9235de4a14a10e3ed05c91f0a34f25a7` / tree
  `8b70a30f7310758d797284071256bce9fb80ecb0`. This includes the upstream
  request/KV ownership cleanup from sglang#36958 while retaining the GLM,
  native-FlashInfer, ReplaySSM, and precompile integration work.
- Retain rc.31's private-stream `:4096:2:16:8` cuBLAS workspace profile and
  advance the compiled-cache schema to `v19`. v0.1.0-rc.32 remains unbuilt and
  unqualified; strict lifecycle diagnostics are a runtime qualification
  profile, not baked into ordinary serving behavior.

## v0.1.0-rc.31

- Pin SGLang integration commit
  `fbe8f3827bdae568d46fc2acce83802c81c22576` / tree
  `1009f52478adb81edd4a95e06322a73c7e2e3f31`. Revert the rc.30 shared DSA
  compression-gate stream after the exact C4 workload caused a CUDA illegal
  memory access and pod restart. Preserve one CUDA stream per indexer layer.
- Set PyTorch's cuBLAS workspace profile to `:4096:2:16:8`, its general
  pre-Blackwell default, while retaining private stream identity. This is an
  exact-hardware C1-through-C4 memory/performance experiment, not a capacity or
  zero-tradeoff claim. Advance the compiled-cache schema to `v18`;
  v0.1.0-rc.31 remains unbuilt and unqualified.

## v0.1.0-rc.30

- Retain v0.1.0-rc.29's exact SGLang, FlashInfer, ModelOpt, model, cache ABI,
  and runtime profile. Correct the build-only draft-sharing lifecycle contract
  to inspect `alloc_memory_pool()`, where the ordinary embedding/output-head
  rebind actually occurs. v0.1.0-rc.29 failed before image publication;
  v0.1.0-rc.30 remains unbuilt and unqualified.

## v0.1.0-rc.29

- Pin SGLang integration commit
  `d34c0b44e3f90f80ccbbe06202cc3387e3728d10` / tree
  `0dad5d76ff0a95b4a2378086005eef7c23dc1ddf`. Reuse one named CUDA stream per
  process/device for the DSA indexers' serialized compression-gate work instead
  of retaining one stream and cuBLAS workspace per layer. Preserve the existing
  within-layer overlap and synchronization contract.
- Remove the unproductive early draft embedding/output-head alias attempt after
  exact-hardware measurement showed no steady-state memory saving. Retain the
  ordinary serving alias without changing any tensor value. Advance the
  compiled-cache schema to `v17`; this candidate remains unbuilt and
  unqualified.

## v0.1.0-rc.28

- Retain v0.1.0-rc.27's exact SGLang, FlashInfer, ModelOpt, model, and runtime
  profile. Correct the build-time ReplaySSM contract check to inspect the
  kernel's Python module because Triton exposes the decorated kernel as a
  `JITFunction`, not a plain Python function. v0.1.0-rc.27 failed before image
  publication; v0.1.0-rc.28 remains unbuilt and unqualified.

## v0.1.0-rc.27

- Pin SGLang integration commit
  `0a107d8f74da4621a09f5b498b10fd366b839ad4` / tree
  `02830060c888a9a84330ba15966d4c444ddc867e`. Rebind the draft embedding and
  output head to the target immediately after loading, before memory-pool
  sizing, without changing tensor values or quantization.
- Detect ReplaySSM capability from GLM's KDA cache contract and include the
  fused KDA verify ring-write work from upstream pull request 36821. This keeps
  ReplaySSM opt-in pending exact-candidate qualification.
- Precompile W4A16 route-pack specializations for 256, 320, 512, 1,024, 2,048,
  4,096, and 8,192 token prefills before pool allocation. Advance the compiled
  cache schema to `v16`; retain the exact model, quantization, vision, parsers,
  native FlashInfer DSA, and adaptive MTP profile from v0.1.0-rc.26. This
  candidate is not built or qualified.

## v0.1.0-rc.26

- Pin SGLang integration commit
  `6ab3d299fda185969c601c58430804dff09c253c` / tree
  `9e96af7cfdf4b18911a6647d23b8b25919852247`. Correct native-NoPE detection
  to use the actual 512-wide absorbed query instead of GLM-5.3's configured
  256-wide pre-absorption dimension, restoring the required temporary
  2,051-to-2,176 `-1` index padding.
- Pin FlashInfer integration commit
  `7cbd1aecd7581137f3b18dfbb4f47b09957dc7cf` / tree
  `6a957df7b48adac53ac27d2156b46bc2455ce157`. Pack native GLM NoPE KV rows
  to their exact 528-byte payload while retaining wider padded-row decode
  compatibility; this changes no KV values or scale precision.
- Select native FlashInfer for prefill and decode and advance the compiled-cache
  schema to `v15`. Retain the exact quantized model, vision path, BF16 MTP
  non-expert tensors, adaptive 5/1/6 speculation, parsers, and C4 graph profile.
  This candidate is not qualified until exact-hardware evidence exists.

## v0.1.0-rc.25

- Rebase SGLang onto upstream main `24c9251ac52ada1660f372922c72c1d3af722247`
  and pin integration commit `835e4579bce3c7c01015f3e288840005561c2d64` /
  tree `d63a350fa932d4667dfc674596f6f8c8f4163645`.
- Rebase FlashInfer onto upstream main
  `e425c7b029ca90d5d01ff207913b070863d35a5b` and pin integration commit
  `37550dc84dba16accc2f611b793598c73b39b9ab` / tree
  `abf62cd1561943670473e0b2b151607076138e1b`.
- Compile the supported SM120 BF16 KDA short- and long-prefill paths, FP8 DSA
  index-prefix gather, and W4A16 route pack before runtime memory-pool sizing.
  Static KDA winners are keyed by capability, dtype, geometry, and semantic
  flags; unsupported shapes retain ordinary autotuning.
- Advance the compiled-cache schema to `v14`. Retain the exact model artifact,
  quantization, vision, native adaptive MTP, parsers, and TileLang control from
  v0.1.0-rc.24. This candidate is not qualified until exact-hardware stability,
  memory, capacity, vision, and performance evidence is recorded.

## v0.1.0-rc.24

- Scope the native FlashInfer SM120 sparse-MLA validator to configurations
  that actually select that backend. The preceding guard incorrectly rejected
  the independent raw-FP8 TileLang pair during attention-backend construction.
- Add a CPU regression covering TileLang pass-through and mixed native/backend
  rejection. Default the launcher to the upstream-reviewed raw-FP8 TileLang
  path while retaining native FlashInfer for isolated qualification.
- Retain v0.1.0-rc.23's exact model, quantization, FlashInfer, ModelOpt, vision,
  MTP, and cache-schema bytes. This candidate is not qualified until exact
  hardware evidence is recorded.

## v0.1.0-rc.23

- Correct the build-time contract for FlashInfer's public
  `SparseMLASm120Wrapper` alias: verify its callable `run` API instead of
  requiring the aliased implementation class to rewrite Python's internal
  `__name__` attribute.
- Retain the exact v0.1.0-rc.22 SGLang, FlashInfer, ModelOpt, native NoPE,
  W4A16, vision, MTP, and cache-layout bytes. This is a packaging assertion
  correction, not a runtime, model-quality, capacity, or performance change.

## v0.1.0-rc.22

Image build failed at the final packaging assertion; no image was published.

- Advance SGLang to `6a0b9caba5b324bc7d52f976c02f6ae57b116dea` /
  tree `6a49f92e7f2ade242845a1054845a5ea4ec8f1b0` and FlashInfer to
  `be0d04071cec17666bed9940109228caeab23911` / tree
  `ccabb6e13093ad7d0101a640a99ff296cfe6d133`.
- Integrate FlashInfer pull request 4802's native SM120 no-RoPE sparse-MLA
  kernel for GLM-5.3. Preserve all 2,051 model candidates by padding with
  `-1` to the 2,176-entry physical contract and use a zero-initialized
  656-byte scaled-FP8 cache row with a 528-byte meaningful prefix.
- Retain one persistent FlashInfer runner and carve small-batch decode scratch
  from SGLang's existing workspace. Keep the proven in-place W4A16 prepared
  weights, exact ModelOpt quantization contract, vision, and native MTP.
- Select `flashinfer_sparse_mla` for both prefill and decode and bump the cache
  schema to `v13`. This is a source candidate; performance, capacity, and
  quality remain unqualified until exact-hardware evidence is recorded.

## v0.1.0-rc.21

- Advance SGLang to `6e5844d435838c68ab5c2c78b5b49c4136a541de` / tree
  `7c5a1f0aaab0c4bca3497e592c95dccef7c2baf6` after v0.1.0-rc.20 proved
  that FlashInfer's functional W4A16 path retained a second packed expert bank
  for every layer and exhausted HBM during target-only warmup.
- Prepare FP4 expert weights in their byte-identical source allocations, reuse
  the original K32 scale buffers, retain prepared views on each layer, and
  dispatch through FlashInfer's prepared-weight path.
- Fail closed if prepared weights are missing and add CPU storage-contract
  tests. Retain the exact FlashInfer, ModelOpt, and cache-schema pins; no model
  tensors or quantization numerics change.

## v0.1.0-rc.20

- Advance SGLang to `5b6297ce555a03c79a3dfec691bb2e1cdf70708c` / tree
  `f8be560548744a22bc585f0160c3149daee9fb70` so GLM's native draft retains
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
- Select the one-stage BF16 TileLang DSA schedule when the GPU exposes less
  than 120 KiB of opt-in shared memory per block. This keeps GLM's NoPE decode
  kernel below SM120's limit while retaining the two-stage datacenter default.
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
