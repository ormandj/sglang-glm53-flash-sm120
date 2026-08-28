# Benchmarks

**`v0.1.0-rc.14` is not built or qualified. `v0.1.0-rc.13` failed served-model
correctness and has no performance results.** The direct BF16 to MXFP4 artifact
is complete and hash-verified, and the exact v0.1.0-rc.13 image boots with the
requested 524,288-token target pool. Its target-only outputs still echo or
repeat prompts instead of answering them. v0.1.0-rc.14 adds an independent
SM120-safe TileLang launch for a BF16-KV correctness control; only exact-image
evidence can qualify it.

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
| target KV format | 528 bytes/token/layer: 512 E4M3 latent bytes with four arbitrary FP32 scales and no RoPE payload | corrected exact vendor model/config/cache-path audit; the earlier 584-byte row inspected an unused DSv4-specific layout |
| fused KPool vendor preimage | all 4 draft-PR regressions fail, including 100% mismatch for exact ties and coarse-bin overflow | exact v0.1.0-rc.5 image on SM120; [`evidence/v0.1.0-rc.5-kpool-topk-regressions-20260828.txt`](evidence/v0.1.0-rc.5-kpool-topk-regressions-20260828.txt) |
| artifact-free MXFP4 source JIT | repaired vendor suite 4/4 passed; prior config-ownership source 3/4 | exact v0.1.0-rc.5 image plus the two-line rc.8 diagnostic delta on SM120; [`evidence/v0.1.0-rc.5-no-cubin-sm120-hardware-20260828.txt`](evidence/v0.1.0-rc.5-no-cubin-sm120-hardware-20260828.txt) |
| artifact-free sparse MLA source JIT | 4/4 production-shaped decode/prefill and sink/no-sink cases passed | exact v0.1.0-rc.5 image and pinned FlashInfer test on SM120; same evidence file |
| native target + MTP load | both target ranks loaded in 477.8s and native layer-45 MTP loaded in 8.16s; warmup then failed closed on the unsupported inherited KPool guard | exact v0.1.0-rc.8 image; pre-candidate startup diagnostic, not served qualification; [`evidence/v0.1.0-rc.8-kpool-adapter-diagnostic-20260828.txt`](evidence/v0.1.0-rc.8-kpool-adapter-diagnostic-20260828.txt) |
| prior DSv4 KPool experiment | invalidated for GLM-5.3: it numerically tested a real DSv4 kernel, but the model never uses that 584-byte cache; the v0.1.0-rc.9 SGLang branch was dead and warmup still rejected KPool | source-path and runtime-geometry correction after v0.1.0-rc.9 startup; not evidence for v0.1.0-rc.12 |
| v0.1.0-rc.11 unfused KPool diagnostic | target weights and a 524,288-token FP8 KV pool loaded; warmup then failed the generic transform's exact-2,048 assertion on GLM's 2,051-entry table | identifies a transform-contract defect before inference; not output-quality evidence; [`evidence/v0.1.0-rc.11-startup-diagnostic-20260828.txt`](evidence/v0.1.0-rc.11-startup-diagnostic-20260828.txt) |
| v0.1.0-rc.12 target-only diagnostic | the corrected 2,051-entry transform booted, but a deterministic marker repeated `742`; production-path reproduction then proved the no-RoPE writer overwrote reserved physical slot 0 | candidate not qualified; [`evidence/v0.1.0-rc.12-startup-diagnostic-20260828.txt`](evidence/v0.1.0-rc.12-startup-diagnostic-20260828.txt) |
| v0.1.0-rc.13 target-only diagnostic | the reserved-slot contract passed and output became coherent, but chat and raw completions echoed/repeated their prompts; both short and >2,048-token controls were wrong | candidate not qualified; native MTP deliberately not enabled; [`evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt`](evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt) |
| v0.1.0-rc.13 Marlin MoE control | with the exact checkpoint, TP2/NCCL, FP8 cache, sparse-MLA backends, vision, parsers, and 524,288-token pool unchanged, the independent Marlin MXFP4 path produced the same prompt echo/repetition | rules out the FlashInfer MXFP4 MoE runner as the served failure's root cause; no performance claim; [`evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt`](evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt) |
| v0.1.0-rc.13 eager control | with prefill and decode CUDA graphs disabled and the exact checkpoint, MoE/DSA backends, TP2/NCCL, FP8 cache, vision, parsers, and 524,288-token pool unchanged, chat still echoed and raw completion repeated zeros | rules out CUDA-graph capture and replay as the served failure's root cause; no performance claim; [`evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt`](evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt) |
| v0.1.0-rc.13 TRT-LLM DSA control | the B200/GB300-style FP8-KV TRT-LLM selection loaded the model and 524,288-token pool, then rc13's SM120 guard rejected `trtllm` before backend construction | no inference result; generic TRT-LLM MLA remains unsupported on SM120 in pinned FlashInfer 0.6.18; [`evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt`](evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt) |
| v0.1.0-rc.13 actual-weight MXFP4 GPU diagnostic | real layers 3/23/44 agree between decompressed quant and the exact SM120 kernel at cosine 0.9989..0.9990; TP2 split/reduce agrees with the full kernel at cosine 0.999993; wrong gate/up controls fall to 0.579..0.648 | rules out the principal packed-weight/runtime ABI hypothesis, but BF16-to-quant one-layer output relative L2 remains 0.198..0.217 and full-model quality is still unqualified; [`evidence/v0.1.0-rc.13-actual-mxfp4-gpu-20260828.txt`](evidence/v0.1.0-rc.13-actual-mxfp4-gpu-20260828.txt) |

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

Every v0.1.0-rc.14 qualification row must identify the full image name
`v0.1.0-rc.14`, image digest, model
artifact manifest hash, exact launcher arguments, GPU/driver state, raw evidence
path, and pass/fail criterion. Do not import performance numbers from another
model, another quant, B200/GB300, or a different candidate.
