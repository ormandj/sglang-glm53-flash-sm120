# GLM-5.3-Flash SM120 container

This repository builds the immutable runtime used by the primary
`sglang-glm53-flash-sm120` qualification repository.

Current candidate:
`git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.21`.
Local build name: `sglang-glm53-flash-sm120:v0.1.0-rc.21`.

**v0.1.0-rc.21 is a source candidate, not a qualified release.** Performance,
quality, context, vision, and MTP results belong in the primary repository with
exact-candidate evidence.

The vendor base is pinned by its linux/amd64 OCI manifest and supplies the known
CUDA/PyTorch environment only. Its unverifiable SGLang tarball is shadowed by
the exact SGLang integration tree recorded in `stack.lock.json`. FlashInfer and
ModelOpt are also installed from exact commits and tree hashes.

This candidate adds load-time ownership of FlashInfer's prepared W4A16 layout.
The FP4 weights are tiled into their byte-identical checkpoint allocation, the
prepared K32 scales replace their equally sized source buffers, and dispatch
uses retained prepared views instead of a second process-lifetime weight cache.
No model tensor values or quantization choices change.

The preceding GLM NextN correction remains included. The
inherited DeepSeek draft constructor normally clears ModelOpt FP4 because its
native draft is BF16, while GLM may serialize the layer-45 routed experts as
FP4. The config is now preserved only for that GLM case; a checkpoint-declared
whole-layer ignore still selects BF16. Cache schema `v12` prevents reuse of
draft graphs built against the previous loader contract.

The exact v0.1.0-rc.19 FlashInfer TC-decode replay fix remains pinned. Its
auto-selected constrained `K=32/N=512` FC2 tile is accepted by the same exact
predicate during custom-op replay.

This build intentionally contains none of the rc.16-and-earlier MXFP4,
no-RoPE, TileLang shared-memory, sentinel, or CPU-offload patch stack. Current
source includes the legitimate upstream fixes, while the new E4M3-K32 W4A16
contract is isolated in the pinned SGLang and FlashInfer integration commits.

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
  -t sglang-glm53-flash-sm120:v0.1.0-rc.21 .
```

The Forgejo release workflow refuses to overwrite an existing SemVer candidate
tag. A successful image build makes this candidate built, not qualified.
