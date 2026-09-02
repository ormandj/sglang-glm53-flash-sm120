# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs (SM120).
#
# The vendor image supplies the CUDA/PyTorch dependency stack only. Its
# SGLang tree has no git provenance, so this candidate puts an exact, verifiable
# SGLang integration tree first on PYTHONPATH and rebuilds FlashInfer from exact
# official bases plus checksummed project patches. No rc.14 vendor-byte patches
# are carried forward.
ARG GLM53_RELEASE_VERSION=0.1.1
ARG GLM53_RELEASE_CANDIDATE=14
ARG GLM53_CACHE_SCHEMA=v54
ARG GLM53_SGLANG_BASE=lmsysorg/sglang@sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_BASE_TAG=glm-5.3-flash
ARG GLM53_SGLANG_BASE_INDEX=sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST=sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_REPOSITORY=https://github.com/sgl-project/sglang.git
ARG GLM53_SGLANG_HEAD=4c2c169e6ba15aee5408b250ce25ff7e73388d9b
ARG GLM53_SGLANG_UPSTREAM_TREE=3babfd36d877df796283b229eb96145e7846750f
ARG GLM53_SGLANG_TREE=9b2b368a6b0ce6726980dab7ac784e1f9f980e36
ARG GLM53_SGLANG_PATCH_SHA256=010a3c9348d4d981cc71878d43a640894177f3803fa81b1aa8c686fcf20e72d0
ARG GLM53_FLASHINFER_REPOSITORY=https://github.com/flashinfer-ai/flashinfer.git
ARG GLM53_FLASHINFER_VERSION=0.6.18
ARG GLM53_FLASHINFER_HEAD=286eee4e2999a825716eab68e597cb1ee0881e1b
ARG GLM53_FLASHINFER_UPSTREAM_TREE=ff920cf35c77ffba88aabf19fc3fe815dd998d93
ARG GLM53_FLASHINFER_TREE=bdc33c22f0e0c8e16321890ed3ebb2b82f4afdce
ARG GLM53_FLASHINFER_PATCH_SHA256=05f3b12ae578cb2bfec2951de94c3f63d2f8afc6eb7eed613ca5ebea80db97f6
ARG GLM53_MODELOPT_REPOSITORY=https://github.com/NVIDIA/Model-Optimizer.git
ARG GLM53_MODELOPT_VERSION=0.47.0rc0
ARG GLM53_MODELOPT_RELEASE_TAG=0.47.0rc0
ARG GLM53_MODELOPT_HEAD=022767c7ab3d7d36211affd85e5c496770cde768
ARG GLM53_MODELOPT_TREE=9ccd3a130aec6a0ebcf85f6a0dd724b50d0e8bd9
ARG GLM53_MODEL_REPOSITORY=zai-org/GLM-5.3-Flash-BF16
ARG GLM53_MODEL_REVISION=f12e0fe1f6b2ea274c11a569582edfd99d993c5e
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

FROM ${GLM53_SGLANG_BASE} AS runtime
ARG GLM53_RELEASE_VERSION
ARG GLM53_RELEASE_CANDIDATE
ARG GLM53_CACHE_SCHEMA
ARG GLM53_SGLANG_BASE_TAG
ARG GLM53_SGLANG_BASE_INDEX
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST
ARG GLM53_SGLANG_REPOSITORY
ARG GLM53_SGLANG_HEAD
ARG GLM53_SGLANG_UPSTREAM_TREE
ARG GLM53_SGLANG_TREE
ARG GLM53_SGLANG_PATCH_SHA256
ARG GLM53_FLASHINFER_REPOSITORY
ARG GLM53_FLASHINFER_VERSION
ARG GLM53_FLASHINFER_HEAD
ARG GLM53_FLASHINFER_UPSTREAM_TREE
ARG GLM53_FLASHINFER_TREE
ARG GLM53_FLASHINFER_PATCH_SHA256
ARG GLM53_MODELOPT_REPOSITORY
ARG GLM53_MODELOPT_VERSION
ARG GLM53_MODELOPT_RELEASE_TAG
ARG GLM53_MODELOPT_HEAD
ARG GLM53_MODELOPT_TREE
ARG GLM53_MODEL_REPOSITORY
ARG GLM53_MODEL_REVISION
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

ENV SGLANG_SOURCE_ROOT=/opt/sglang-source \
    PYTHONPATH=/opt/sglang-source/python \
    FLASHINFER_NO_DOWNLOAD=1 \
    FLASHINFER_CUDA_ARCH_LIST=12.0f \
    CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8

# Project-owned integration deltas remain in this internal build repository.
# Each patch is applied to an exact official-upstream tree and the resulting
# complete source tree is verified before it can shadow the vendor tree.
COPY patches/sglang-glm53-integration.patch /tmp/sglang-glm53-integration.patch
COPY patches/flashinfer-glm53-integration.patch /tmp/flashinfer-glm53-integration.patch

# Replace the unverifiable vendor Python tree with the exact patched SGLang
# integration tree. The delta contains GLM-5.3 support, native FlashInfer SM120
# NoPE sparse MLA, and the E4M3-K32 W4A16 loader/runner contract.
RUN set -eux; \
    git init -q "${SGLANG_SOURCE_ROOT}"; \
    cd "${SGLANG_SOURCE_ROOT}"; \
    git remote add origin "${GLM53_SGLANG_REPOSITORY}"; \
    git fetch --depth=1 origin "${GLM53_SGLANG_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${GLM53_SGLANG_HEAD}"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "${GLM53_SGLANG_UPSTREAM_TREE}"; \
    printf '%s  %s\n' "${GLM53_SGLANG_PATCH_SHA256}" /tmp/sglang-glm53-integration.patch | sha256sum -c -; \
    git apply --check /tmp/sglang-glm53-integration.patch; \
    git apply --index /tmp/sglang-glm53-integration.patch; \
    test "$(git write-tree)" = "${GLM53_SGLANG_TREE}"; \
    rm /tmp/sglang-glm53-integration.patch; \
    uv run --no-project --python /opt/sglang/bin/python python -m compileall -q \
      python/sglang/srt/models/glm5_next.py \
      python/sglang/srt/models/glm5_next_nextn.py \
      python/sglang/srt/models/deepseek_nextn.py \
      python/sglang/srt/speculative/eagle_worker_v2.py \
      python/sglang/srt/arg_groups/model_hook.py \
      python/sglang/kernels/ops/attention/fla/chunk_intra.py \
      python/sglang/kernels/ops/attention/fla/kda.py \
      python/sglang/kernels/ops/attention/fla/fused_kda_conv_recurrent_verify.py \
      python/sglang/kernels/ops/attention/flash_mla_sm120.py \
      python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py \
      python/sglang/kernels/ops/layernorm/mhc.py \
      python/sglang/srt/layers/attention/dsa_backend.py \
      python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py \
      python/sglang/srt/layers/attention/linear/kda_backend.py \
      python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py \
      python/sglang/srt/mem_cache/kv_cache_configurator.py \
      python/sglang/srt/mem_cache/allocator/paged.py \
      python/sglang/srt/mem_cache/unified_radix_cache.py \
      python/sglang/srt/mem_cache/unified_cache/components/full_component.py \
      python/sglang/srt/utils/async_probe.py \
      python/sglang/srt/layers/quantization/modelopt_quant.py \
      python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py

# Build FlashInfer from the exact SM120 integration tree. It contains
# upstream's native NoPE sparse-MLA SM120 kernels, >INT32 W4A16 expert-bank
# correction, exact TC-decode tile replay, and the explicit ModelOpt
# E2M1/E4M3-K32 preparation contract. Generic cubin bundles are deliberately
# removed so this image cannot silently select a kernel built for another GPU.
RUN set -eux; \
    git init -q /tmp/flashinfer-source; \
    cd /tmp/flashinfer-source; \
    git remote add origin "${GLM53_FLASHINFER_REPOSITORY}"; \
    git fetch --depth=1 origin "${GLM53_FLASHINFER_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${GLM53_FLASHINFER_HEAD}"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "${GLM53_FLASHINFER_UPSTREAM_TREE}"; \
    printf '%s  %s\n' "${GLM53_FLASHINFER_PATCH_SHA256}" /tmp/flashinfer-glm53-integration.patch | sha256sum -c -; \
    git apply --check /tmp/flashinfer-glm53-integration.patch; \
    git apply --index /tmp/flashinfer-glm53-integration.patch; \
    test "$(git write-tree)" = "${GLM53_FLASHINFER_TREE}"; \
    git submodule update --init --recursive --depth=1; \
    uv pip uninstall --python /opt/sglang/bin/python \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache || true; \
    BUILD_NVEP=0 FLASHINFER_CUDA_ARCH_LIST=12.0f \
      uv pip install --python /opt/sglang/bin/python --no-deps .; \
    cd /; \
    rm -rf /tmp/flashinfer-source /tmp/flashinfer-glm53-integration.patch

# The runtime image doubles as the controlled quantization environment. Pin
# current ModelOpt source instead of relying on the base image's older wheel.
RUN set -eux; \
    git init -q /tmp/modelopt-source; \
    cd /tmp/modelopt-source; \
    git remote add origin "${GLM53_MODELOPT_REPOSITORY}"; \
    git fetch --depth=1 origin "${GLM53_MODELOPT_HEAD}"; \
    git fetch --depth=1 origin \
      "refs/tags/${GLM53_MODELOPT_RELEASE_TAG}:refs/tags/${GLM53_MODELOPT_RELEASE_TAG}"; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${GLM53_MODELOPT_HEAD}"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "${GLM53_MODELOPT_TREE}"; \
    test "$(git rev-parse "${GLM53_MODELOPT_RELEASE_TAG}^{commit}")" = "${GLM53_MODELOPT_HEAD}"; \
    uv pip install --python /opt/sglang/bin/python --reinstall --no-deps .; \
    cd /; \
    rm -rf /tmp/modelopt-source

# The ModelOpt HF base-model loader imports accelerate; the vendor tree omits
# it, which crashed the first draft-model load under a modelopt_mixed target
# (DFLASH drafter, 2026-08-31). Pinned, no-deps: every dependency it needs is
# already present.
RUN uv pip install --python /opt/sglang/bin/python --no-deps accelerate==1.12.0

# Fail the image build if any part of the intended contract is shadowed by the
# vendor tree or omitted from the installed packages.
COPY acceptance_check.py /tmp/acceptance_check.py
RUN set -eux; \
    CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG} GLM53_FLASHINFER_HEAD=${GLM53_FLASHINFER_HEAD} GLM53_FLASHINFER_VERSION=${GLM53_FLASHINFER_VERSION} GLM53_MODELOPT_VERSION=${GLM53_MODELOPT_VERSION} \
    uv run --no-project --python /opt/sglang/bin/python python /tmp/acceptance_check.py

ENV SGLANG_BUILD_COMMIT=${GLM53_SGLANG_HEAD} \
    SGLANG_BUILD_TREE=${GLM53_SGLANG_TREE} \
    FLASHINFER_VERSION=${GLM53_FLASHINFER_VERSION} \
    FLASHINFER_BUILD_COMMIT=${GLM53_FLASHINFER_HEAD} \
    MODELOPT_BUILD_COMMIT=${GLM53_MODELOPT_HEAD}

LABEL org.opencontainers.image.title="sglang-glm53-flash-sm120" \
      org.opencontainers.image.description="GLM-5.3-Flash E4M3-K32 W4A16 integration for dual RTX PRO 6000 Blackwell" \
      org.opencontainers.image.source=${IMAGE_SOURCE} \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version=${GLM53_RELEASE_VERSION} \
      org.opencontainers.image.revision=${IMAGE_SOURCE_REVISION} \
      ai.release.candidate=rc.${GLM53_RELEASE_CANDIDATE} \
      ai.release.cache-schema=${GLM53_CACHE_SCHEMA} \
      ai.hardware.target-architecture="sm120" \
      ai.model.repository=${GLM53_MODEL_REPOSITORY} \
      ai.model.revision=${GLM53_MODEL_REVISION} \
      ai.model.quantization="planned ModelOpt E2M1 weights/E4M3 K32 scales, W4A16" \
      ai.sglang.repository=${GLM53_SGLANG_REPOSITORY} \
      ai.sglang.head=${GLM53_SGLANG_HEAD} \
      ai.sglang.tree=${GLM53_SGLANG_TREE} \
      ai.sglang.patch-sha256=${GLM53_SGLANG_PATCH_SHA256} \
      ai.flashinfer.repository=${GLM53_FLASHINFER_REPOSITORY} \
      ai.flashinfer.head=${GLM53_FLASHINFER_HEAD} \
      ai.flashinfer.tree=${GLM53_FLASHINFER_TREE} \
      ai.flashinfer.patch-sha256=${GLM53_FLASHINFER_PATCH_SHA256} \
      ai.modelopt.repository=${GLM53_MODELOPT_REPOSITORY} \
      ai.modelopt.release-tag=${GLM53_MODELOPT_RELEASE_TAG} \
      ai.modelopt.head=${GLM53_MODELOPT_HEAD} \
      ai.modelopt.tree=${GLM53_MODELOPT_TREE}
