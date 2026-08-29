# GLM-5.3-Flash SM120 engineering record

This is the durable investigation record for bringing GLM-5.3-Flash up on two
RTX PRO 6000 Blackwell GPUs. It deliberately includes failed candidates,
disproven hypotheses, measurements from controls, and open questions. It is not
a substitute for the exact-candidate qualification receipts in
`BENCHMARKS.md` and `evidence/`.

## Non-negotiable target

- two RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs, TP=2 and EP=1;
- the locally produced high-quality ModelOpt W4A16 artifact;
- vision enabled and exercised with a real image;
- native adaptive EAGLE/NextN MTP, nominally 5 steps, top-k 1, and 6 draft
  tokens;
- `glm45` reasoning parsing and `glm47` tool-call parsing;
- FP8 E4M3 KV;
- a shared active-token pool near 500K if the measured safe memory envelope
  permits it;
- C4 for one user's agentic coding fanout, with C8 treated as a desirable burst
  experiment rather than a reason to reduce model quality;
- high decode and prefill performance without CPU weight offload.

The optimization order is correctness, quality, vision, native MTP, stability,
memory efficiency, and then speed within the surviving configurations. Weight
precision is not reduced merely to make an early runtime fit.

## Hardware budget and units

The preflight receipt measured each card as 97,886 MiB total, including 637 MiB
reserved by the device, and 97,248 MiB free before model load. The pair exposed
195,772 MiB total and 194,496 MiB free. These are MiB values converted with
binary units; they must not be mixed with decimal GB printed by some SGLang
logs. Runtime OOM reports describe each process-visible card as approximately
94.97 GiB.

## Checkpoint and quantization decisions

The rejected local MXFP4 artifact is not a serving candidate. It produced
prompt echo/repetition and corrupt output through independent MoE and attention
controls. A real-weight diagnostic showed that its kernel agreed with its own
dequantized representation, but the representation was already about 0.20
relative-L2 from BF16 expert weights. That separates an artifact-quality
failure from the runtime ABI failures found later.

The replacement artifact was produced directly from
`zai-org/GLM-5.3-Flash-BF16` revision
`f12e0fe1f6b2ea274c11a569582edfd99d993c5e` with ModelOpt
`0.47.0rc0` at `022767c7ab3d7d36211affd85e5c496770cde768`.
Its contract is:

- routed expert weights only: signed E2M1, packed two per byte;
- E4M3FN scales per K=32 group plus FP32 global scales;
- BF16 activations, so this is W4A16 rather than W4A4;
- ModelOpt's 126-candidate static-MSE scale search;
- shared experts, routers, vision, embeddings/head, attention, DSA, KDA/mHC,
  normalization, and convolutions remain source precision;
- native layer 45 MTP routed experts use the same W4A16 recipe as target routed
  experts. The remaining MTP modules remain BF16, matching the target policy.

The completed artifact contains 184,905,778,296 tensor bytes (172.2069 GiB),
150,226 tensors, and 93 shards. It quantized 37,152 routed source projections
and preserved 1,618 source-precision tensors, including 347 vision tensors.
The quantization job took 1,357.64 seconds, or about 22 minutes 38 seconds, on
local SM120 hardware.

Artifact-level reconstruction was:

- aggregate expert relative-L2: 0.0851127164;
- mean matrix relative-L2: 0.0850405739;
- maximum matrix relative-L2: 0.1006871065;
- minimum matrix cosine: 0.9949183464.

These values establish packing and reconstruction integrity, not end-to-end
model quality. BF16-teacher logit KLD, top-1 agreement, task quality, vision,
long-context retrieval, and MTP acceptance remain distinct qualification gates.

K=32 was chosen because K=16 scale storage would add roughly 9.07 GiB across
the artifact. K=16 should ordinarily preserve more weight information, so the
K=32 choice is an explicit capacity tradeoff, not a quality claim.

### MTP precision and the rejected extra-quantization idea

The draft is not a complete BF16 copy. Layer 45's 288 routed experts are already
E2M1/E4M3-K32 W4A16. Its attention/indexer, `eh_proj`, shared experts, router,
normalization, and output support remain BF16. The serialized layer-45 payload
is about 3.9302 GiB, of which about 3.5859 GiB is already quantized routed
experts. The remaining plausibly quantizable BF16 linears amount to only about
0.216 GiB per TP rank. Even an aggressive FP4 conversion would save only about
0.16 GiB per rank, around 0.32 GiB across the pair, buying approximately 16K to
22K shared tokens on the measured TileLang layout. This is not worth the
quality and kernel risk, so no further MTP quantization is planned.

## Runtime failures and what they established

### SM120 TileLang dynamic shared memory

An early BF16 TileLang DSA decode schedule requested 169,984 bytes of dynamic
shared memory and was rejected by SM120. A one-stage, lower-shared-memory
schedule is a legitimate hardware correction. A blanket shared-memory patch is
not retained now that current raw-FP8 TileLang dispatch selects an SM120-safe
launch.

### ModelOpt NextN loader contract

The inherited DeepSeek NextN logic normally discards ModelOpt FP4 for the draft.
That is correct when a checkpoint declares its complete draft layer ignored,
but wrong for this artifact because layer 45 routed experts are serialized
W4A16. The GLM loader must preserve ModelOpt FP4 only when checkpoint metadata
actually quantizes those draft modules. This is a loader/checkpoint contract
fix, not a change to model values.

### W4A16 prepared-weight duplication

The v0.1.0-rc.20 target-only run loaded 84.71 GB per rank and had 9.26 GB
available before pools. FlashInfer's functional W4A16 preparation path then
retained a second process-lifetime packed expert bank and OOMed while requesting
another 576 MiB. This was not KV exhaustion. The correct implementation
prepares the byte-identical source allocation in place, reuses K32 scale
storage, retains prepared views on the layer, and dispatches the prepared path.
That removed the duplicate expert bank without changing a weight or kernel
result.

### Recurrent-state slots are real

This hybrid model consumes five recurrent KDA/Mamba slots per live request with
the selected native MTP profile. C4 therefore requires 20 slots. Replacing a
padding sentinel with a live slot can silently corrupt another request and is
not a valid memory optimization. The slot count is a design constraint unless
upstream changes the state representation or speculation algorithm.

### Late Triton compilation caused the long-request crash

The v0.1.0-rc.24 service reached ready state and passed meaningful C1 and C4
requests, but a later Pi request with an 8,192-token prefill caused previously
unseen Triton specializations and autotune candidates to compile after all
runtime pools and CUDA graphs had been committed. Free memory fell from about
2.39 GB per rank to about 0.35 GB. Allocations of 20 MiB and then 256 MiB failed.

The late set included KDA intra/inter/recompute and state-update kernels, DSA
index-prefix gather, and W4A16 route-pack specializations. The 65,536-token KV
pool did not itself exhaust memory. The fix is to compile the bounded set of
expected short- and long-prefill specializations before memory-pool allocation,
then size pools from the post-compile baseline. Compiled caches should be
persisted in a candidate- and ABI-specific PVC subpath so a restart does not
repeat the work.

Model-specific kernel pins are not the intended upstream form. Static winners
are keyed by device capability, dtype, tensor geometry, and semantic flags.
The GLM model hook supplies its shape; generic kernel code decides whether a
curated specialization exists and otherwise retains upstream autotuning.

## v0.1.0-rc.24 control measurements

The exact image was
`git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v0.1.0-rc.24@sha256:c7ff501b55b9931cf8a2af1bfad11120d0c5a21680bd67d69bc5e65da8fd4379`.
It used raw-layout FP8 TileLang DSA, FlashInfer CUTLASS W4A16 MoE, Triton KDA,
vision, adaptive MTP 5/1/6, `glm45`/`glm47`, C4/20 recurrent slots, batch 1-4
decode graphs, and a deliberately small 65,536-token pool.

Observed load/pool values per rank were approximately:

- target model: 84.71 GB;
- native draft load increment: 3.19 GB;
- target FP8 KV at 65,536 tokens: 0.43 GB;
- draft KV: 0.04 GB;
- free after decode graph capture: about 2.39 GB.

The raw TileLang DSA accounting is 7,728 bytes per shared token per rank for the
selected twelve physical DSA layers: a 512-byte latent row plus a 132-byte
index row per layer. This is the current capacity baseline and must be
remeasured if either backend changes its cache ownership or row layout.

Before the late-compile crash:

- C1, 4,096 requested output tokens, reasoning enabled: 11.52 s TTFT,
  118.28 reported decode token/s after first delta, 46.15 s total latency;
- C4, four 2,048-token responses: 1.19 s mean TTFT, 279.97 aggregate client
  token/s, about 75.77 mean per-request decode token/s, and server-side decode
  peaks around 383.86 token/s;
- adaptive MTP moved between 3/1/4 and 5/1/6 as acceptance changed; observed
  accept length rose to roughly 3.15 and accept rate to roughly 0.72 in a
  favorable part of the C4 run;
- a real Pi-to-AISIX-to-SGLang request returned coherent output, proving the
  configured gateway path before the later memory failure.

These are useful controls, not final claims: the candidate crashed under a
larger prefill, the output-length stops do not establish task correctness, and
vision/tool/quality/long-context gates were not all completed on that exact
candidate.

## TileLang and FlashInfer DSA A/B

TileLang is the current stable control because its raw FP8 path boots and
produces strong measured decode throughput. FlashInfer is still the likely
long-term SM120 path, but its native 656-byte wrapper and scratch ownership must
be corrected and measured rather than assumed superior.

The A/B must use identical model bytes, parsers, vision, MTP, chunk size, graph
set, concurrency, and prompts. Record:

- post-load, post-precompile, post-pool, and post-graph free HBM per rank;
- exact persistent and temporary bytes owned by the DSA backend;
- effective shared-token bytes per rank and maximum safe pool;
- prefill throughput and TTFT on genuinely large prompts;
- C1 and C4 decode throughput and MTP acceptance;
- stability across the first large request and a restart using persisted
  compiled artifacts;
- real-image and tool-call correctness.

FlashInfer wins only if corrected layout/ownership yields a measured capacity,
performance, or stability advantage. Backend selection alone must not alter
quantization or vision support.

## Configuration lessons

- Use Argo CD, not Flux. Repository manifests remain scaled to zero; manual
  scale-up/scale-down is allowed for qualification.
- Parsers are explicit: `--reasoning-parser glm45` and
  `--tool-call-parser glm47`. Auto-detected `deepseek-r1` reasoning or `glm45`
  tool parsing is not the desired contract.
- TP=2 uses EP=1. Expert parallelism does not help on this two-GPU PCIe setup.
- Shared-expert fusion stays disabled because protected shared experts are BF16
  while routed experts are serialized FP4.
- Vision requires multimodal enablement and the PIL image processor; CPU image
  preprocessing is acceptable because preserving GPU model/KV capacity is more
  valuable than moving image preprocessing onto GPU.
- Large prompts are mandatory for prefill and memory validation. Tiny queries
  can hide uncompiled kernels and give unrealistic throughput.
- Never expose server arguments in evidence because they may contain an API
  key. Filter `server_args`, `api_key`, `--api-key`, and `SGLANG_API_KEY` before
  displaying or saving logs.
- A successful build means built; readiness means booted; coherent text is a
  correctness control. None alone means qualified.

## Open work

1. Finish generic pre-pool compilation of all expected KDA, DSA gather, and
   W4A16 route specializations.
2. Rebase the integration tree onto current SGLang main and retain only fixes
   that still correspond to reproduced failures.
3. Build an immutable next candidate with an ABI-specific persistent compile
   cache.
4. Repeat the large Pi workload and verify no compile occurs after pool
   commitment and no later memory growth appears.
5. Run the corrected native FlashInfer 656-byte DSA A/B against TileLang.
6. Expand the shared pool from the 65,536-token diagnostic setting to the
   largest value that leaves repeatable headroom at C4.
7. Qualify real vision, nested tools, long-context retrieval, MTP acceptance,
   BF16-teacher KLD/top-1, and C1/C4 performance on one exact candidate.

