# Quantization contract

The old BF16-to-MXFP4 artifact is rejected. It produced corrupt served output
through independent MoE and attention backends, and actual expert tensors had
roughly 0.20 one-layer relative-L2 error. Its scripts were removed from the
active repository; historical receipts remain under `evidence/`.

The v0.1.0-rc.18 target is a new ModelOpt W4A16 artifact produced directly from
`zai-org/GLM-5.3-Flash-BF16` revision
`f12e0fe1f6b2ea274c11a569582edfd99d993c5e`.

## Selected representation

Only routed-expert gate/up/down weights are quantized initially:

- value format: signed E2M1, two weights packed per uint8;
- activation format: BF16 (W4A16), never FP4/MXFP8 activation quantization;
- group axis: input/K dimension, 32 consecutive values per group;
- block scale: E4M3FN, logical shape `[out, K/32]` per expert projection;
- global weight scale: one FP32 scalar per down projection and one shared FP32
  scalar for each fused gate+up pair;
- input scales: neutral FP32 `1.0`, because the SM120 W4A16 kernel consumes BF16
  activations and its alphas are pure weight-global scales.

Gate and up must be calibrated jointly before splitting so both halves carry
the same `weight_scale_2`. Latest ModelOpt explicitly preserves this invariant;
violating it is a known way to get coherent-looking but wrong MoE output.

The checkpoint metadata is ModelOpt's flat schema:

```json
{
  "quant_method": "modelopt",
  "quant_algo": "W4A16_NVFP4",
  "group_size": 32,
  "kv_cache_scheme": null,
  "producer": {
    "name": "modelopt",
    "version": "0.47.0rc0",
    "commit": "022767c7ab3d7d36211affd85e5c496770cde768"
  }
}
```

The final `ignore` list is derived from the published GLM ModelOpt layout and
must protect the vision stack, embeddings/LM head, attention and KDA/mHC
linears, routers, shared experts, and any dense MLP. Routed `*.mlp.experts.*`
weights are the inclusion set. Selection is validated against tensor names and
byte counts before any output is published; a zero-match or unexpected-match
pattern is fatal.

## Static MSE calculation

For each BF16 expert matrix (or joint gate+up matrix), let `A` be its global
absolute maximum. Latest pinned ModelOpt performs a 126-candidate E4M3 scale
sweep per K=32 block and returns the block amax minimizing weight reconstruction
MSE. The stored values are:

```text
weight_scale_2 = A / (6 * 448)
weight_scale   = E4M3FN(best_block_amax / A * 448), clamped to [2^-9, 448]
weight          = pack_E2M1(original / (weight_scale * weight_scale_2))
```

`6` is the largest E2M1 magnitude and `448` is the E4M3FN normalization maximum.
All-zero blocks use ModelOpt's nonzero scale sentinel before E4M3 clamping; the
packed values remain exactly zero.
The producer calls pinned ModelOpt's `nvfp4_fp8_scale_sweep` and
`NVFP4QTensor.quantize(..., block_size=32, try_tensorrt=False)` rather than a
locally approximated rounding implementation.

The audited producer is
`quantization/quantize_glm53_bf16_w4a16_e4m3_k32.py`. Its exact-GPU preflight
checks all 38,770 tensors across 120 source shards, including 38 gate/up pairs
that cross a source-shard boundary and must still be calibrated jointly. See
the [exact-GPU preflight receipt](evidence/v0.1.0-rc.18-glm-quant-preflight-20260828.txt).

## Protected tensors and staged sensitivity

The first valid artifact keeps these in BF16:

- visual encoder, merger/projector, and all multimodal support tensors;
- token embeddings and LM head;
- routers/gates and shared experts;
- attention, DSA indexer, KDA, mHC, normalization, and convolution tensors;
- all non-expert layer-45 MTP tensors.

Layer-45 MTP routed experts use the same W4A16 recipe. Retaining the complete
draft expert bank in BF16 would consume roughly ten extra GiB and defeat the KV
target. Any later FP8 attention experiment is a separate artifact and may be
accepted only if it passes teacher KLD and all modality/long-context gates.
Because the shared experts remain BF16, serving must use
`--disable-shared-experts-fusion`; otherwise their BF16 tensors would be mapped
into a serialized FP4 fused-expert allocation.

## Output and size gates

The producer writes to a temporary sibling directory and publishes atomically
only after all checks pass. It records source revision, ModelOpt/SGLang/
FlashInfer commits, selection counts, tensor dtypes/shapes, per-component byte
totals, per-shard hashes, and the complete quantization config.

The exact preflight projection from the pinned BF16 index is 184,905,778,296
bytes (172.2069 GiB): 37,152 routed source weights become packed E2M1 plus
E4M3-K32 and scalar scales, while 19,339,524,984 bytes of non-routed tensors
remain source-exact. This is still a projection until the output artifact is
written and hashed. Against 189.9404 GiB of measured physical memory across the
pair it leaves 17.7335 GiB total before runtime pools. The artifact must still
prove that real loader packing, FP8 KV, C4 recurrent state, MTP workspaces, and
CUDA graphs fit without CPU offload.

K32 stores half as many scales as the public K16 control and saves about 9.07
GiB overall, but K16 normally has a quality advantage when calibration is held
constant. The K32 choice is therefore a capacity tradeoff, not a quality claim;
the pinned MSE recipe and BF16 protected tensors must pass the teacher gates.

## Quality gates

Packed-kernel agreement is necessary but not sufficient. Qualification compares
the final artifact to the pinned BF16 teacher with speculation disabled:

1. per-layer and end-to-end logit KLD over every causal position, repeated over
   representative code, reasoning, multilingual, tool, and visual prompts;
2. top-1 token agreement and explicit tail/outlier inspection;
3. deterministic text and nested `glm47` tool calls;
4. real-image understanding with the vision token path observed in logs;
5. long-context retrieval at multiple depths through the intended pool size;
6. MTP-on output consistency plus proposed/accepted-token statistics.

No threshold is retrofitted after seeing results. BF16 receipts and the public
g16 checkpoint are controls; neither is evidence for our g32 artifact.
