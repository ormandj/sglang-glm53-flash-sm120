# GLM-5.3-Flash SM120 container

This repository builds the immutable runtime used by the primary
`sglang-glm53-flash-sm120` qualification repository.

Current candidate:
`git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.18`.
Local build name: `sglang-glm53-flash-sm120:v0.1.0-rc.18`.

**v0.1.0-rc.18 is a source candidate, not a qualified release.** Performance,
quality, context, vision, and MTP results belong in the primary repository with
exact-candidate evidence.

The vendor base is pinned by its linux/amd64 OCI manifest and supplies the known
CUDA/PyTorch environment only. Its unverifiable SGLang tarball is shadowed by
the exact SGLang integration tree recorded in `stack.lock.json`. FlashInfer and
ModelOpt are also installed from exact commits and tree hashes.

This candidate additionally fetches and verifies ModelOpt's exact
`0.47.0rc0` tag at the pinned commit so `setuptools-scm` emits the locked
package version. Runtime source trees and cache schema are unchanged from rc.17.

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
  -t sglang-glm53-flash-sm120:v0.1.0-rc.18 .
```

The Forgejo release workflow refuses to overwrite an existing SemVer candidate
tag. A successful image build makes this candidate built, not qualified.
