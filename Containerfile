# Exact upstream sources and checked integration patches; CUDA/PyTorch from the pinned base.
ARG GLM53_RELEASE_VERSION=0.5.0
ARG GLM53_RELEASE_CANDIDATE=4
ARG GLM53_CACHE_SCHEMA=v88
ARG GLM53_SGLANG_BASE=lmsysorg/sglang@sha256:7a9ef6dc376369247e1dd9bbabc4f3d02a60ca8bc7253574f70d99fb28cbe9a0
ARG GLM53_SGLANG_BASE_TAG=nightly-dev-cu13-20260928-81f27fb3
ARG GLM53_SGLANG_BASE_INDEX=sha256:7a9ef6dc376369247e1dd9bbabc4f3d02a60ca8bc7253574f70d99fb28cbe9a0
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST=sha256:7a9ef6dc376369247e1dd9bbabc4f3d02a60ca8bc7253574f70d99fb28cbe9a0
ARG GLM53_SGLANG_REPOSITORY=https://github.com/sgl-project/sglang.git
ARG GLM53_SGLANG_HEAD=c7be3e935b5034006cd6ae7977b41e2459b4126c
ARG GLM53_SGLANG_UPSTREAM_TREE=754d67baa8f9d3c9c0f6b2aa5f112b826b90eacc
ARG GLM53_SGLANG_TREE=d6ba0d49a19ab3b5b12312fc35d08a4c1b95e93e
ARG GLM53_SGLANG_PATCH_SHA256=f876e842e466eb75759e4b5c59e4eb097eed9abc2f131e4eb6c683dfa1acae4c
ARG GLM53_FLASHINFER_REPOSITORY=https://github.com/flashinfer-ai/flashinfer.git
ARG GLM53_FLASHINFER_VERSION=0.7.0
ARG GLM53_FLASHINFER_HEAD=90a709ca842996e3431ece0346d97b73693eff94
ARG GLM53_FLASHINFER_UPSTREAM_TREE=54328f309052090ac6f02d0cb502d56009d60ff2
ARG GLM53_FLASHINFER_TREE=ece95b67b28a4523e01fc42a0af270f41747aaee
ARG GLM53_FLASHINFER_PATCH_SHA256=1278f2832aa437054dad7edbbcd1c2d4d34db5c78af2ef64415a7b233f0e6f0a
ARG GLM53_MODELOPT_REPOSITORY=https://github.com/NVIDIA/Model-Optimizer.git
ARG GLM53_MODELOPT_VERSION=0.47.0.dev0+gitc2aaa44f
ARG GLM53_MODELOPT_HEAD=c2aaa44f6040658a21a2f5d2213c10ecc6542512
ARG GLM53_MODELOPT_TREE=96a6310748ac292510260e362c3196ca9bafb211
ARG GLM53_MODEL_REPOSITORY=nvidia/GLM-5.3-Flash-NVFP4
ARG GLM53_MODEL_REVISION=09b04e5e74bca08ca8549fc736d4cdd8624bfde3
ARG GLM53_DEEPGEMM_REPOSITORY=https://github.com/sgl-project/DeepGEMM.git
ARG GLM53_DEEPGEMM_HEAD=c518ae0ab137922333e2d4ff59f77f8bc6ca4b57
ARG GLM53_DEEPGEMM_TREE=0ee476cce438258aef9a72aef86c246881fbca78

ARG GLM53_TRANSFORMERS_REPOSITORY=https://github.com/huggingface/transformers.git
ARG GLM53_TRANSFORMERS_HEAD=1e71827951aaebd23e3da65ad33f2eebfb970367
ARG GLM53_TRANSFORMERS_TREE=fd0770dda26861b65f558cd216cbaa219df8e187
ARG GLM53_TRANSFORMERS_VERSION=5.18.0.dev0

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
    SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev \
    && rm -rf /var/lib/apt/lists/*
COPY stack.lock.json /opt/glm53/stack.lock.json
COPY patches/sglang-glm53-integration.patch /opt/glm53/patches/sglang-glm53-integration.patch
COPY patches/flashinfer-glm53-integration.patch /opt/glm53/patches/flashinfer-glm53-integration.patch
COPY scripts/build-stack.sh scripts/prepare-stack.py /opt/glm53/scripts/
COPY acceptance_check.py /opt/glm53/acceptance_check.py
RUN bash /opt/glm53/scripts/build-stack.sh
LABEL org.opencontainers.image.title="GLM-5.3-Flash on SM120" \
      org.opencontainers.image.description="NVIDIA NVFP4 target and native BF16 MTP on four RTX PRO 6000 GPUs" \
      org.opencontainers.image.version="0.5.0-rc.4" \
      org.opencontainers.image.source=${IMAGE_SOURCE} \
      org.opencontainers.image.revision=${IMAGE_SOURCE_REVISION} \
      ai.hardware.target-architecture="sm120" \
      ai.release.cache-schema="v88"
