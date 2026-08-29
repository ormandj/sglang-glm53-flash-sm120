# GLM-5.3-Flash SM120 container

This repository builds the immutable runtime used by the primary
`sglang-glm53-flash-sm120` qualification repository.

Current candidate:
`git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.40`.
Local build name: `sglang-glm53-flash-sm120:v0.1.0-rc.40`.

**v0.1.0-rc.40 is a source candidate, not a qualified release.** Performance,
quality, context, vision, and MTP results belong in the primary repository with
exact-candidate evidence.

The vendor base is pinned by its linux/amd64 OCI manifest and supplies the known
CUDA/PyTorch environment only. Its unverifiable SGLang tarball is shadowed by
the exact SGLang integration tree recorded in `stack.lock.json`. FlashInfer and
ModelOpt are also installed from exact commits and tree hashes.

This candidate makes native FlashInfer the qualification path. It fixes the
GLM adapter's geometry check so the model's configured 256-wide pre-absorption
NoPE dimension cannot suppress the required 2,051-to-2,176 temporary index
padding for the actual 512-wide absorbed query.

The preceding draft embedding and output-head serving alias remains in its
upstream lifecycle position. An earlier attempt to move it before pool sizing
did not change the measured steady-state footprint and is intentionally absent.
The candidate recognizes GLM's KDA cache contract as ReplaySSM-capable and
includes the upstream fused KDA verify ring-write path. ReplaySSM remains an
explicit qualification A/B rather than a launcher default.

The candidate compiles the supported SM120 BF16 KDA short- and long-prefill
specializations, FP8 DSA index-prefix gather, and W4A16 route-pack regimes from
256 through 8,192 prefill tokens before runtime memory pools are sized. The
candidate preserves one compression-gate CUDA stream per DSA indexer layer. A
shared-stream experiment saved allocator memory but failed the exact C4 test
with an illegal memory access and is explicitly reverted. Instead, this
candidate tests PyTorch's pre-Blackwell `:4096:2:16:8` cuBLAS workspace profile
while retaining private stream identity. That profile is not a free-memory
claim until exact C1-through-C4 performance and correctness evidence exists.
Static KDA tactics are keyed by capability,
dtype, tensor geometry, and semantic flags rather than a GLM model name;
unrecognized shapes retain ordinary autotuning. This prevents the first large
request from loading compiler candidates after KV, recurrent-state, and CUDA
graph allocations have consumed the memory envelope.

The image also integrates FlashInfer's native SM120 sparse-MLA kernel for
GLM-5.3's no-RoPE attention geometry. SGLang pads the model's 2,051 candidates
to the kernel's 2,176-entry physical contract and stores exactly 528
scaled-FP8 bytes per token (512 latent values plus four FP32 scales), with no
reserved RoPE suffix. Decode uses a persistent
FlashInfer runner and reuses SGLang's existing workspace for small-batch
scratch. No model tensor values or quantization choices change.

The preceding load-time ownership of FlashInfer's prepared W4A16 layout remains
included. FP4 weights are tiled into their byte-identical checkpoint allocation,
prepared K32 scales replace equally sized source buffers, and dispatch retains
prepared views instead of a second process-lifetime weight cache.

The preceding GLM NextN correction remains included. The
inherited DeepSeek draft constructor normally clears ModelOpt FP4 because its
native draft is BF16, while GLM may serialize the layer-45 routed experts as
FP4. The config is now preserved only for that GLM case; a checkpoint-declared
whole-layer ignore still selects BF16. Cache schema `v27` prevents reuse of
graphs and JIT objects built against the preceding SGLang, FlashInfer, and
late-compilation behavior.

The candidate also carries debug-gated allocator and unified-radix probes. The
exact v0.1.0-rc.36 diagnostic first observed the bad whole-tree state
synchronously when a finished-request insert completed: eight consecutive
int64 slots contained the paired-int32 pattern `(96, 96)`, while the fresh
insert values and the new-node, unevict, and split boundaries were clean. This
candidate synchronizes around every insert action, maps every reachable Full
value to its node, parent, key length, storage pointer and offset, preserves an
exact snapshot across each action, and rejects any allocator free whose byte
range overlaps a reachable Full value. The checks are inactive in ordinary
serving. This is fault localization, not a claimed runtime fix.

The exact v0.1.0-rc.19 FlashInfer TC-decode replay fix remains pinned. Its
auto-selected constrained `K=32/N=512` FC2 tile is accepted by the same exact
predicate during custom-op replay.

This build intentionally contains none of the v0.1.0-rc.16-and-earlier MXFP4,
vendor-byte, sentinel, or CPU-offload patch stack. Native NoPE support derives
from FlashInfer pull request 4802 and its SGLang adapter is isolated in exact,
verifiable integration commits. The E4M3-K32 W4A16 serving contract remains
isolated in those same pinned source trees.

Verify before pushing:

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

Build locally:

```bash
podman build \
  --target runtime \
  --build-arg IMAGE_SOURCE=https://git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-glm53-flash-sm120:v0.1.0-rc.40 .
```

The Forgejo release workflow refuses to overwrite an existing SemVer candidate
tag. A successful image build makes this candidate built, not qualified.
