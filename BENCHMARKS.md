# Benchmarks

**`v0.1.0-rc.9` has no served-model quality or performance results yet.** The
direct BF16 to MXFP4 artifact is complete and hash-verified; the corrected image
is still being built. A candidate is built, not qualified, until exact-candidate
evidence is recorded here.

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
| production artifact | 131 files; 184,945,092,190 bytes; all 37,152 routed weights serialized; 1,618/1,618 protected tensors bit-exact | artifact manifest `a74810a1…`; [`evidence/quantization-artifact-20260828.txt`](evidence/quantization-artifact-20260828.txt) |
| DFlash2 artifact | 2,342,175,855 bytes, block 8, window 2,048, capture layers 5/14/24/33/42 | immutable HF revision `7d74cdd…` |
| SGLang patch preimages | vendor `mxfp4.py` and `glm5_next.py` match exact pinned SHA-256 values | vendor image digest only; no git-provenance claim |
| physical GPU framebuffer | 95.592 GiB/card; 191.184 GiB/pair; 189.938 GiB free before model load | DCGM/driver 595.71.05; [`evidence/preflight-gpu-memory-20260828.txt`](evidence/preflight-gpu-memory-20260828.txt) |
| target KV format | 584 bytes/token/layer: 448 E4M3 NOPE bytes with seven per-token E8M0 scales, 64 BF16 RoPE values, one scale pad byte | exact pinned vendor source; [`evidence/v0.1.0-rc.5-fp8-kv-source-audit-20260828.txt`](evidence/v0.1.0-rc.5-fp8-kv-source-audit-20260828.txt) |
| fused KPool vendor preimage | all 4 draft-PR regressions fail, including 100% mismatch for exact ties and coarse-bin overflow | exact v0.1.0-rc.5 image on SM120; [`evidence/v0.1.0-rc.5-kpool-topk-regressions-20260828.txt`](evidence/v0.1.0-rc.5-kpool-topk-regressions-20260828.txt) |
| artifact-free MXFP4 source JIT | repaired vendor suite 4/4 passed; prior config-ownership source 3/4 | exact v0.1.0-rc.5 image plus the two-line rc.8 diagnostic delta on SM120; [`evidence/v0.1.0-rc.5-no-cubin-sm120-hardware-20260828.txt`](evidence/v0.1.0-rc.5-no-cubin-sm120-hardware-20260828.txt) |
| artifact-free sparse MLA source JIT | 4/4 production-shaped decode/prefill and sink/no-sink cases passed | exact v0.1.0-rc.5 image and pinned FlashInfer test on SM120; same evidence file |
| native target + MTP load | both target ranks loaded in 477.8s and native layer-45 MTP loaded in 8.16s; warmup then failed closed on the unsupported inherited KPool guard | exact v0.1.0-rc.8 image; pre-candidate startup diagnostic, not served qualification; [`evidence/v0.1.0-rc.8-kpool-adapter-diagnostic-20260828.txt`](evidence/v0.1.0-rc.8-kpool-adapter-diagnostic-20260828.txt) |
| GLM DSv4 KPool dual-segment path | one-token decode and 128-token prefill matched the dequantized reference; all three live tail entries survived a 512-entry `-1` gap | exact v0.1.0-rc.8 image plus the v0.1.0-rc.9 adapter on SM120; pre-candidate numerical regression; same evidence file |

The production quantizer repeated and expanded the structural checks, including
bit-for-bit comparison of every protected tensor and 129 layer/projection
round-trip probes. It emitted `.quant-complete` only after those checks, and a
separate publication job re-hashed every copied file before atomic publication.
These checks qualify the artifact structure, not model behavior.

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

Every row must identify the full image name `v0.1.0-rc.9`, image digest, model
artifact manifest hash, exact launcher arguments, GPU/driver state, raw evidence
path, and pass/fail criterion. Do not import performance numbers from another
model, another quant, B200/GB300, or a different candidate.
