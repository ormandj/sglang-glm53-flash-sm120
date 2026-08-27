# Running the candidate

```bash
export IMAGE=sglang-glm53-flash-sm120:v0.1.0-rc.1
export MODEL_DIR=/models/zai-org/GLM-5.3-Flash-MXFP4
export CACHE_DIR=/srv/cache/sglang-glm53-flash-sm120-v1
./examples/serve-glm53-flash.sh
```

`CACHE_DIR` must be **image-specific**. Compiled kernels, TileLang and Triton
caches are not portable across incompatible builds; reusing another image's
cache directory produces confusing runtime failures. The cache schema for this
release is `v1`.

## Serving envelope

TP=2, `fp8_e4m3` KV, `--mem-fraction-static 0.96`, `--max-running-requests 8`,
`--mamba-ssm-dtype bfloat16`, TileLang DSA backends, `glm45`/`glm47` parsers.

Three choices that are load-bearing and easy to get wrong:

- **`--mamba-ssm-dtype bfloat16` is mandatory.** SGLang defaults the SSM state to
  FP32. The 34 KDA layers hold 72.78 MiB of recurrent state per slot, and that
  state is allocated per `--max-running-requests`, not per live request — so the
  default costs ~2.2 GiB of KV pool at 8 slots and ~4.5 GiB at 32.
- **Expert Parallel is deliberately not used.** At TP=2 it saves essentially no
  VRAM while adding ~27 % expected routing imbalance at C=1.
- **MTP defaults off.** sglang #36653 and #36599 both block NEXTN for this exact
  configuration (TP>1, FP4 draft). Set `NEXTN=1` only to test a fix.

## First boot

Confirm from the log that the allocated KV pool is the **512-wide MLA latent**
cache. A decompressed MHA fallback would be 360,448 B/token — a ~20x capacity
loss that reads like a tuning problem rather than a wrong code path.

## Diagnostics

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/models | jq .
curl -s localhost:8000/metrics | grep -E 'token_usage|cache_hit'
```
