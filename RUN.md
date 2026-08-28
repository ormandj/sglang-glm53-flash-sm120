# Running `v0.1.0-rc.13`

```bash
export IMAGE=git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.13
export MODEL_DIR=/models/zai-org/GLM-5.3-Flash-BF16-MXFP4
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v9
export SPECULATIVE_MODE=mtp
./examples/serve-glm53-flash.sh
```

Cache schema `v9` is mandatory because the no-RoPE KV writer now protects the
reserved padding slot and applies DCP ownership mapping. Do not reuse compiled
`v8` artifacts.

`SPECULATIVE_MODE` accepts:

- `mtp`: native layer-45 NEXTN via fixed five-step EAGLE;
- `dflash`: pinned DFlash2 assistant;
- `none`: verifier-only control.

For DFlash2, also set:

```bash
export SPECULATIVE_MODE=dflash
export DFLASH_DIR=/models/incoai/GLM-5.3-Flash-DFlash2
```

All modes keep vision enabled with CPU image preprocessing, use TP=2, pin
`flashinfer_mxfp4`, and disable shared-expert fusion to preserve its BF16 path.
The capacity-first launcher uses one live request slot, five recurrent-state
slots, batch-one CUDA graphs, and PyTorch's default allocator. Confirm the
exact image digest, successful target/draft weight loads, multimodal
initialization, selected all-reduce path, and allocated token pools from startup
logs. Qualification results belong only in the primary repository.
