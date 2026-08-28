# Benchmarks

**`v0.1.0-rc.4` has no model-quality or performance results yet.** The BF16 to
MXFP4 artifact and corrected image are still being built. A candidate is built,
not qualified, until exact-candidate evidence is recorded here.

## Pre-candidate checks completed

These checks chose the design and validated tooling. They are not a served-model
qualification:

| check | result | scope |
|---|---|---|
| BF16 source manifest | 120 safetensors shards; 642,646,653,816 indexed tensor bytes | immutable HF revision `f12e0fe1…` |
| source tensor namespace | 38,770 tensors | metadata inspection |
| routed-expert target | 37,152 tensors = 43 x 288 x 3 | exact anchored-name audit |
| protected scope | 1,618 tensors, including 347 vision tensors | exact namespace complement |
| serialized format | `mxfp4-pack-quantized`, GROUP, group 32, uint8 packed values and E8M0 scales | synthetic GLM-shaped checkpoint |
| synthetic round-trip | relative L2 0.112915 | toolchain serialization only; not production weights |
| source choice | direct BF16 source has lower sampled reconstruction error than FP8-to-MXFP4 | tensor probes only; see `QUANTIZATION.md` |
| DFlash2 artifact | 2,342,175,855 bytes, block 8, window 2,048, capture layers 5/14/24/33/42 | immutable HF revision `7d74cdd…` |
| SGLang patch preimages | vendor `mxfp4.py` and `glm5_next.py` match exact pinned SHA-256 values | vendor image digest only; no git-provenance claim |
| physical GPU framebuffer | 95.592 GiB/card; 191.184 GiB/pair; 189.938 GiB free before model load | DCGM/driver 595.71.05; [`evidence/preflight-gpu-memory-20260828.txt`](evidence/preflight-gpu-memory-20260828.txt) |

The production quantizer repeats and expands the structural checks, including
bit-for-bit comparison of every protected tensor and 129 layer/projection
round-trip probes. Its results will be added only after `.quant-complete` and
artifact-hash validation succeed.

## Required qualification matrix

### Load and memory

- target-only, native MTP, and DFlash2 boot at TP=2;
- exact per-GPU model residency;
- KDA recurrent-state allocation at eight request slots;
- target and draft KV bytes/token;
- allocated pooled tokens, with 500,000 as the goal;
- CUDA graph and free-memory headroom.

### Text quality

- deterministic smoke and instruction following;
- code/tool-call and reasoning-parser behavior;
- token-level top-1 agreement and KL against the BF16 source on representative
  traces;
- greedy divergence and error growth with position through long KDA contexts;
- repeated-tool prompts relevant to open SGLang issue #36669.

### Vision quality

- image-dependent description, OCR, and multi-image isolation;
- deterministic comparison with the BF16 source where feasible;
- repeated requests relevant to active multimodal-isolation work.

### Performance

- time to first token and output tokens/second for `none`, `mtp`, and `dflash`;
- concurrency 1, 2, 4, and 8 where memory permits;
- speculative acceptance length/tokens per step;
- short prompt, long prefill, and first decode after cold prefills above 262k;
- custom PCIe all-reduce versus NCCL if both paths are stable.

Every row must identify the full image name `v0.1.0-rc.4`, image digest, model
artifact manifest hash, exact launcher arguments, GPU/driver state, raw evidence
path, and pass/fail criterion. Do not import performance numbers from another
model, another quant, B200/GB300, or a different candidate.
