# Running `v0.1.0-rc.5`

```bash
export IMAGE=sglang-glm53-flash-sm120:v0.1.0-rc.5
export MODEL_DIR=/models/zai-org/GLM-5.3-Flash-BF16-MXFP4
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v4
export SPECULATIVE_MODE=mtp
./examples/serve-glm53-flash.sh
```

`CACHE_DIR` must be image-specific. Compiled FlashInfer, TorchInductor, TileLang,
and Triton artifacts are not portable across incompatible candidates.
v0.1.0-rc.5 uses cache schema `v4` because its SM120 mHC/indexer fallback
contract differs from v0.1.0-rc.4.

## Serving envelope

The launcher pins TP=2, vision enabled, compressed-tensors MXFP4 routed experts,
FP8 E4M3 KV, TRT-LLM DSA, BF16 KDA recurrent state, eight request slots, and a
524,288-token configured context ceiling. The actual pooled-token capacity is a
startup measurement, not the value of `--context-length`.

Three settings are load-bearing:

- Keep `--mamba-ssm-dtype bfloat16`. FP32 roughly doubles the 34 KDA layers'
  per-slot recurrent state and silently removes memory from KV.
- Do not enable Expert Parallel at TP=2. It does not materially reduce expert
  residency and adds PCIe all-to-all plus routing imbalance.
- Keep `--moe-runner-backend flashinfer_mxfp4`. The inherited automatic choice
  does not safely cover this compressed-tensors GLM artifact, and the separate
  `flashinfer_trtllm` path has an open out-of-bounds routing issue.
- Keep `--disable-shared-experts-fusion`. The shared expert is intentionally
  bit-exact BF16 and must not be appended to the MXFP4 routed-expert buffer.

Custom all-reduce is left enabled. SGLang tests whether the two PCIe GPUs have
working P2P and falls back to NCCL if not. Capture the selected path in evidence;
do not force either result without an A/B.

## Speculative modes

Native MTP is the default and is not optional in the final qualification:

```bash
export SPECULATIVE_MODE=mtp
```

It uses the checkpoint's quantized layer 45 through SGLang's EAGLE alias for
NEXTN, with adaptive 5-step, top-k 1, six-draft-token bounds.

Run a verifier-only control with:

```bash
export SPECULATIVE_MODE=none
```

Run DFlash2 with:

```bash
export SPECULATIVE_MODE=dflash
export DFLASH_DIR=/models/incoai/GLM-5.3-Flash-DFlash2
```

The DFlash2 directory must be the pinned revision in `stack.lock.json`. Its
draft uses block size 8, a 2,048-token logical window, FA4 attention, and FP8
draft KV. FP8 draft KV changes proposal efficiency, not verifier weights; output
quality and acceptance still require measurement.

## First boot gates

Do not treat a listening port as a successful boot. Preserve the complete log
and verify:

1. both target ranks load the `mxfp4-pack-quantized` artifact without missing,
   unexpected, or shape-mismatched tensors;
2. `flashinfer_mxfp4` selects the SM120 MXFP8-by-MXFP4 path;
3. the target cache is the 512-wide MLA latent plus DSA indexer state, not a
   decompressed MHA fallback;
4. the allocated token pool is at least 500,000 if the goal is met;
5. MTP loads layer 45 through compressed-tensors on both TP ranks;
6. DFlash2 captures layers 5, 14, 24, 33, and 42 and reports its separate draft
   pool when that mode is selected;
7. multimodal initialization remains enabled.

If 500,000 pooled tokens do not fit at `MEM_FRACTION=0.96`, record exact weight,
graph, recurrent-state, and cache allocations before changing the fraction or
request-slot count. Do not trade away vision or MTP to make the log look better.

## Health and text request

```bash
curl -fsS localhost:8000/health
curl -fsS localhost:8000/v1/models | jq .
curl -fsS localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-sm120","messages":[{"role":"user","content":"Return exactly: ready"}],"temperature":0,"max_tokens":16}' | jq .
```

## Vision request

Use a controlled local image or immutable test URL and preserve both the input
hash and response. The OpenAI-compatible content shape is:

```json
{
  "model": "glm53-flash-sm120",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "https://example.invalid/immutable-test.png"}},
      {"type": "text", "text": "Describe the image and read all visible text."}
    ]
  }],
  "temperature": 0,
  "max_tokens": 256
}
```

Replace the placeholder URL; a request that never exercises image embeddings is
not a vision qualification.

## Measurements required before qualification

- verifier-only, MTP, and DFlash2 output throughput and latency at useful
  concurrencies;
- speculative acceptance and accepted tokens per step;
- actual model residency, target pool, draft pool, and maximum admitted context;
- text and vision correctness, plus token-level comparison with the BF16 source;
- long-context behavior and the first decode after cold prefills above 262k;
- repeated tool-calling prompts because relevant upstream failures are open.

Put results and evidence paths in `BENCHMARKS.md`. Do not promote
`v0.1.0-rc.5` from a successful build alone.
