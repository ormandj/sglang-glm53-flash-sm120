# Qualification status

**`v0.1.0-rc.18` is not built and not qualified.** It has no served-model
correctness, quality, capacity, or performance claim yet.

## Required v0.1.0-rc.18 evidence

Every row must record the complete image name and digest, model artifact path
and manifest hash, SGLang/FlashInfer/ModelOpt commits, exact hardware, command
profile with secrets removed, and raw evidence file.

| Gate | Intended test | Status |
|---|---|---|
| source bundle | reproduce exact SGLang, FlashInfer, ModelOpt trees and base manifest | pending |
| image build | imports resolve to pinned SGLang and expose E4M3-K32 W4A16 APIs | pending |
| Qwen canary quant | known-good 35B BF16 MoE: packed vs dequantized vs SM120 kernel | pending |
| Qwen canary serve | deterministic text, tool call, real image, C4 | pending |
| GLM artifact audit | tensor selection/counts, formats, byte totals, hashes, atomic publish | pending |
| GLM target-only | small-pool deterministic text/tool/vision with MTP off | pending |
| GLM native MTP | adaptive 5/1/6 correctness and acceptance statistics | pending |
| quality | BF16-teacher KLD, top-1, task and visual controls | pending |
| context | retrieval probes through the measured active token pool | pending |
| C4 | four simultaneous agent-like requests with 20 recurrent slots | pending |
| C8 burst | eight requests with 40 recurrent slots, queued behavior documented | pending |
| performance | prefill, target decode, MTP effective decode, latency, memory headroom | pending |

The qualification target is vision plus native MTP on TP=2/EP=1, roughly
524,288 total active tokens, and C4. C8 is desirable burst capacity, not a reason
to weaken the quant or silently shrink the pool.

## Controls already established

These controls used earlier candidates or different checkpoints. They guide the
new work but cannot qualify v0.1.0-rc.18.

| Control | Result | Evidence |
|---|---|---|
| dual-GPU physical memory | both RTX PRO 6000 cards expose 96,497 MiB each | [preflight](evidence/preflight-gpu-memory-20260828.txt) |
| Qwen3.5-35B-A3B BF16 platform | exact text, tool, image, and C4 requests passed on the rc.16 dependency stack | cluster receipt pending import into this repository |
| old local MXFP4 artifact | corrupt prompt echo/repetition survived independent MoE and attention controls | [rc.13 target](evidence/v0.1.0-rc.13-target-diagnostic-20260828.txt), [rc.14 TileLang](evidence/v0.1.0-rc.14-tilelang-target-diagnostic-20260828.txt) |
| old actual-weight MXFP4 kernel ABI | kernel and dequantized-quant agreed, but BF16-to-quant expert relative-L2 remained about 0.20 | [actual-weight diagnostic](evidence/v0.1.0-rc.13-actual-mxfp4-gpu-20260828.txt) |
| old TileLang crash | SM120 rejected a 169,984-byte dynamic-shared-memory launch | [rc.14 diagnostic](evidence/v0.1.0-rc.14-tilelang-target-diagnostic-20260828.txt) |

Current SGLang raw-layout FP8 TileLang DSA uses an SM12x launch that fits the
hardware limit, so the old blanket shared-memory patch is not carried forward.
Likewise the runtime's five recurrent slots per request are retained as a real
hybrid-model allocation, while the invalid `-1`-to-live-slot workaround is not.

## Claims discipline

A successful container build makes `v0.1.0-rc.18` built. A health endpoint makes
it booted. Coherent text makes one correctness control pass. None of those alone
qualifies quality, vision, MTP, 500K capacity, C4/C8, or performance.
