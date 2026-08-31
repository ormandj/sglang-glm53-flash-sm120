# Reproducing the published checkpoint

The main [`README.md`](README.md) downloads the completed public checkpoint.
This document is only for reproducing that artifact from the pinned BF16
source:

[`ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO`](https://huggingface.co/ormandj/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO)

## Representation

| Component | Precision |
|---|---|
| Routed experts, layers 3–45 including the MTP expert bank | NVFP4 W4A16, K=32 group scales, MSE-swept |
| Eligible attention, KDA, MLA, and shared-expert projections in layers 0–44 | FP8 E4M3 weight-only, 128x128 block scales |
| Layer-45 non-expert MTP weights, DSA indexer, vision tower and projector, embeddings, LM head, routers, norms, and ineligible linears | Source precision |

The producer deliberately keeps the DSA indexer projections and MTP draft
layer attention in source precision. Quantizing the latter loads through the
draft layer's remapped namespace, where mixed-precision resolution can fail
silently and collapse speculative acceptance even when final outputs remain
correct.

The published artifact contains 177,995,252,856 tensor bytes (165.8 GiB) in
90 safetensors shards. Its model card and included manifests record the exact
selection, reconstruction measurements, producer pins, and per-shard hashes.

## Requirements

- Linux x86_64 with Docker and the NVIDIA Container Toolkit.
- One SM120 GPU.
- Roughly 600 GB for the BF16 source plus 170 GB for the output.
- The Hugging Face CLI available as `hf`.

## Build

Download the exact source revision:

```bash
export SOURCE_DIR=/srv/models/GLM-5.3-Flash-BF16
mkdir -p "$SOURCE_DIR"
hf download zai-org/GLM-5.3-Flash-BF16 \
  --revision f12e0fe1f6b2ea274c11a569582edfd99d993c5e \
  --local-dir "$SOURCE_DIR"
printf 'f12e0fe1f6b2ea274c11a569582edfd99d993c5e\n' \
  > "$SOURCE_DIR/.source-revision"
touch "$SOURCE_DIR/.download-complete"
```

Run the fail-closed producer and publish atomically into the objective output
directory:

```bash
export OUTPUT_DIR=/srv/models/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
docker run --rm --gpus device=0 \
  -v /srv/models:/scratch \
  -v "$PWD/quantization:/opt/q:ro" \
  --entrypoint /opt/sglang/bin/python \
  ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.1.0-rc.67 \
  /opt/q/quantize_glm53_bf16_w4a16_k32_fp8attn_mix.py \
  --source /scratch/GLM-5.3-Flash-BF16 \
  --output /scratch/GLM-5.3-Flash-W4A16-NVFP4-K32-Experts-FP8-WO
```

Publication should be treated as a separate gated operation. Compare the
generated config, index, and all shard hashes to the manifests in the public
repository before uploading or serving the result.
