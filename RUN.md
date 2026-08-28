# Running `v0.1.0-rc.8`

```bash
export IMAGE=git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.8
export MODEL_DIR=/models/zai-org/GLM-5.3-Flash-BF16-MXFP4
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v5
export SPECULATIVE_MODE=mtp
./examples/serve-glm53-flash.sh
```

Cache schema `v5` is mandatory because the patched SM120 sparse-MLA and KPool
sources differ from v0.1.0-rc.5. The later parser and post-loader corrections
do not change compiled kernel sources.

`SPECULATIVE_MODE` accepts:

- `mtp`: native layer-45 NEXTN via adaptive EAGLE;
- `dflash`: pinned DFlash2 assistant;
- `none`: verifier-only control.

For DFlash2, also set:

```bash
export SPECULATIVE_MODE=dflash
export DFLASH_DIR=/models/incoai/GLM-5.3-Flash-DFlash2
```

All modes keep vision enabled, use TP=2, pin `flashinfer_mxfp4`, and disable
shared-expert fusion to preserve its BF16 path. Confirm the
exact image digest, successful target/draft weight loads, multimodal
initialization, selected all-reduce path, and allocated token pools from startup
logs. Qualification results belong only in the primary repository.
