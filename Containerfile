# Exact upstream sources and checked integration patches; CUDA/PyTorch from the pinned base.
ARG GLM53_RELEASE_VERSION=0.6.0
ARG GLM53_RELEASE_CANDIDATE=2
ARG GLM53_CACHE_SCHEMA=v90
ARG GLM53_SGLANG_BASE=lmsysorg/sglang@sha256:53efab7e8255a47b19edc81a0f1ad37cd6c8341bfe8f448d268700b182b2f03d
ARG GLM53_SGLANG_BASE_TAG=nightly-dev-cu13-20261010-961bcf48
ARG GLM53_SGLANG_BASE_INDEX=sha256:e03dd310dc72e64ac709ceb244a79f20fd5e05448c811ab33cc777d8040061ff
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST=sha256:53efab7e8255a47b19edc81a0f1ad37cd6c8341bfe8f448d268700b182b2f03d
ARG GLM53_SGLANG_REPOSITORY=https://github.com/sgl-project/sglang.git
ARG GLM53_SGLANG_HEAD=3831e7e0918052be50ef38b3ff62154573e213d7
ARG GLM53_SGLANG_UPSTREAM_TREE=eb4fd3d60d53b7ea11eb100c29fd0bd110d7cb1c
ARG GLM53_SGLANG_TREE=641f9463c20bd5adcaf05167cbd6a5ea4d6f6715
ARG GLM53_SGLANG_PATCH_SHA256=8fa286edc371b0e7b31eec28ff6ab27d4f19b1f1439b11a735bfe338d84f59e5
ARG GLM53_FLASHINFER_REPOSITORY=https://github.com/flashinfer-ai/flashinfer.git
ARG GLM53_FLASHINFER_VERSION=0.7.2
ARG GLM53_FLASHINFER_HEAD=e9c723a90888a9523c190c779860c19f0f7b7c0f
ARG GLM53_FLASHINFER_UPSTREAM_TREE=6ac577cb7b81b1c233ea9723de35228cf655af42
ARG GLM53_FLASHINFER_TREE=2d60e8e269cc7353893d1a3537e49ccdd88ae558
ARG GLM53_FLASHINFER_PATCH_SHA256=6852c1cda4410d8ba6723ef92b8fbe33f0391476b36d517afe62a1e762856b32
ARG GLM53_MODELOPT_REPOSITORY=https://github.com/NVIDIA/Model-Optimizer.git
ARG GLM53_MODELOPT_VERSION=0.47.0.dev0+gitc2aaa44f
ARG GLM53_MODELOPT_HEAD=c2aaa44f6040658a21a2f5d2213c10ecc6542512
ARG GLM53_MODELOPT_TREE=96a6310748ac292510260e362c3196ca9bafb211
ARG GLM53_MODEL_REPOSITORY=zai-org/GLM-5.3-Flash
ARG GLM53_MODEL_REVISION=eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
ARG GLM53_DEEPGEMM_REPOSITORY=https://github.com/sgl-project/DeepGEMM.git
ARG GLM53_DEEPGEMM_HEAD=cbe1727eacd9ad704795584ce1421dd1f5a8d67e
ARG GLM53_DEEPGEMM_TREE=db497e62dd305d8c461cdd454f732651e8c77265

ARG GLM53_TRANSFORMERS_REPOSITORY=https://github.com/huggingface/transformers.git
ARG GLM53_TRANSFORMERS_HEAD=c4c4605f961877852b89b97478536bb3a26b6fd1
ARG GLM53_TRANSFORMERS_TREE=e3f02e1856e89bef1ea14006b7ba9ca45ac834dc
ARG GLM53_TRANSFORMERS_VERSION=5.19.0

FROM ${GLM53_SGLANG_BASE} AS runtime
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION
ENV PATH=/root/.cargo/bin:/opt/sglang/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONPATH="" \
    FLASHINFER_CUDA_ARCH_LIST=12.0f \
    TVM_FFI_CUDA_ARCH_LIST=12.0 \
    FLASHINFER_NO_DOWNLOAD=1 \
    BUILD_NVEP=0 MAX_JOBS=12 CARGO_BUILD_JOBS=12 \
    SGLANG_RUST_BUILD_MODE=never \
    CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8 \
    SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \
    SGLANG_PCIE_IPC_MAX_NUMEL=786432 \
    SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1 \
    SGLANG_DSA_INDEXER_ROW_SPLIT_MIN_ELEMS=134217728
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev \
    && rm -rf /var/lib/apt/lists/*
COPY stack.lock.json /opt/glm53/stack.lock.json
COPY patches/sglang-glm53-integration.patch /opt/glm53/patches/sglang-glm53-integration.patch
COPY patches/flashinfer-glm53-integration.patch /opt/glm53/patches/flashinfer-glm53-integration.patch
COPY scripts/build-stack.sh scripts/prepare-stack.py /opt/glm53/scripts/
RUN bash /opt/glm53/scripts/build-stack.sh
# Separate layer: acceptance edits must not invalidate the stack build.
COPY acceptance_check.py /opt/glm53/acceptance_check.py
RUN CUDA_VISIBLE_DEVICES=9 /opt/sglang/bin/python /opt/glm53/acceptance_check.py
LABEL org.opencontainers.image.title="GLM-5.3-Flash on SM120" \
      org.opencontainers.image.description="Z.ai FP8 target, FP8 KV and native MTP on four RTX PRO 6000 GPUs" \
      org.opencontainers.image.version="0.6.0-rc.1" \
      org.opencontainers.image.source=${IMAGE_SOURCE} \
      org.opencontainers.image.revision=${IMAGE_SOURCE_REVISION} \
      ai.hardware.target-architecture="sm120" \
      ai.release.cache-schema="v90"
