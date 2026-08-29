# GLM-5.3-Flash SM120 container

This repository builds the immutable runtime used by the primary
`sglang-glm53-flash-sm120` qualification repository.

Current candidate:
`git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.22`.
Local build name: `sglang-glm53-flash-sm120:v0.1.0-rc.22`.

**v0.1.0-rc.22 is a source candidate, not a qualified release.** Performance,
quality, context, vision, and MTP results belong in the primary repository with
exact-candidate evidence.

The vendor base is pinned by its linux/amd64 OCI manifest and supplies the known
CUDA/PyTorch environment only. Its unverifiable SGLang tarball is shadowed by
the exact SGLang integration tree recorded in `stack.lock.json`. FlashInfer and
ModelOpt are also installed from exact commits and tree hashes.

This candidate integrates FlashInfer's native SM120 sparse-MLA kernel for
GLM-5.3's no-RoPE attention geometry. SGLang pads the model's 2,051 candidates
to the kernel's 2,176-entry physical contract and stores its 528 meaningful
scaled-FP8 bytes in a zero-initialized 656-byte row. Decode uses a persistent
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
whole-layer ignore still selects BF16. Cache schema `v13` prevents reuse of
graphs and JIT objects built against the preceding DSA layout and backend.

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
  -t sglang-glm53-flash-sm120:v0.1.0-rc.22 .
```

The Forgejo release workflow refuses to overwrite an existing SemVer candidate
tag. A successful image build makes this candidate built, not qualified.
