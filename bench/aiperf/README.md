# GLM-5.3-Flash benchmark harness

This harness measures an already running OpenAI-compatible server. It does not start or reconfigure the server. [BENCHMARKS.md](../../BENCHMARKS.md) records release measurements, hardware, configuration and limitations.

## Fixed-MTP comparison standard

Run GLM performance comparisons with fixed native MTP: three draft steps, top-k one and four verification tokens, with adaptive switching disabled. Include reasoning tokens in output throughput. The default launcher uses adaptive MTP for serving; [RUN.md](../../RUN.md) explains the benchmark argument changes.

Use `glm-qualification` for the supported four-request profile. It runs five repetitions at each of C1, C2, C3 and C4, plus five cold prefill requests at each of 8k, 32k, 64k and 128k prompt tokens. Higher-concurrency modes in the shared harness do not establish that this model configuration supports those concurrency levels.

The decode workload uses a synthetic coding prompt of approximately 16,384 input tokens, 4,096 output tokens with `ignore_eos=true`, temperature zero, top-p one and a fixed seed panel. Each repetition submits exactly one cohort at its selected concurrency. Analysis covers average context from 17,408 to 20,480 tokens. A window is rejected for prefill activity, queueing, counter resets, incorrect occupancy or insufficient usable context coverage.

Report mean and median output tokens/s after MTP alongside mean and median target forward passes/s, plus output tokens per forward per request, concurrency and the measured window. Output throughput aggregates the concurrent requests. A target forward is a model iteration, not an emitted token. The fixed-output workload includes a post-answer tail that can increase speculative acceptance; it is not completed-answer throughput.

Cold prefill throughput is each request's prompt-token count divided by its time to first token, summarized by mean and median over five requests per length. Those one-output-token probes disable thinking so the streaming content validator can observe the token. This measurement setting does not change serving reasoning policy.

These controlled measurements do not necessarily represent real-world performance. Sequential repetitions within one process do not establish statistical significance or isolate individual patches. Compare identical workloads, seeds, model weights, hardware and analysis definitions, and state any configuration differences.

## Run a panel

AIPerf is pinned by [aiperf.lock.json](aiperf.lock.json). Install that revision with `prepare-in-pod.sh` and the pinned uv binary with `stage-uv-in-pod.sh`, or provide the corresponding executable paths below. Run from this directory inside the serving container, with the CPU-only client environment:

```bash
export CUDA_VISIBLE_DEVICES=9
export MODEL_NAME=glm-5.3-flash
export TOKENIZER_PATH=/models/glm53-flash-w4a16-e4m3-k32
export INFERENCE_URL=http://127.0.0.1:8000
export SERVER_METRICS_URL=http://127.0.0.1:8000/metrics
export BENCH_ENGINE=sglang
export BENCH_DP_SIZE=1
export BENCH_IMAGE_REF='<exact image reference and digest>'
export BENCH_GITOPS_REVISION='<deployment configuration revision>'
export BENCH_PROJECT_REVISION='<benchmark source revision>'
export BENCH_MODEL_REVISION='<model snapshot revision>'
export AIPERF_REVISION=6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23
export AIPERF_PYTHON='<AIPerf environment>/bin/python'
export AIPERF_BIN='<AIPerf environment>/bin/aiperf'
export AIPERF_UV_BIN='<uv installation>/uv'
export AIPERF_CAMPAIGN_ROOT='<persistent result directory>'
./run-engine-gate-in-pod.sh '<campaign>' '<build>' glm-qualification
```

`BENCH_GITOPS_REVISION` is the harness's field name for the deployment configuration revision; the server does not need to be managed by GitOps. Authentication uses `BENCH_API_KEY`, falling back to `SGLANG_API_KEY`. A keyless endpoint needs neither variable. Credential values are excluded from captured environment data.

The engine entry point resolves configuration variables, runs shape warmups and serializes the complete panel. Do not call its `run-in-pod.sh` executor directly for an engine panel. Result directories are immutable. Retain raw metrics, resolved inputs, every valid repetition and analyzer verdicts. A rejected cell cannot support a performance claim.

Keep the server process and configuration unchanged throughout a panel. No other inference, profiler, compiler or GPU test may compete with its measurement window. GPU unit tests require an isolated test process with the serving engine stopped; they must not share a memory-constrained serving process.

## Quality and targeted diagnostics

Quality checks use the adaptive serving configuration separately from the fixed-MTP performance panel. `gsm8k.yaml` runs all 1,319 questions with a 16,384-token output cap. Retain all responses in the denominator, report the answer extraction method and identify output-cap hits. Long-context, image, cache-restoration and tool-response checks exercise different serving behaviors and do not count as additional performance repetitions.

The shared harness also contains turnover and AgentX modes. Select them when the changed path or workload warrants that investigation, with concurrency supported by the tested profile. Their presence does not require unrelated high-concurrency workloads for every GLM release. Optional diagnostics must retain their own definitions and must not replace a failed measurement without a documented reason.
