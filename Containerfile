# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs (SM120).
#
# The vendor image supplies the CUDA/PyTorch dependency stack only. Its
# SGLang tree has no git provenance, so this candidate puts an exact, verifiable
# SGLang integration tree first on PYTHONPATH and rebuilds FlashInfer from an
# exact source tree. No rc.14 MXFP4 or diagnostic patches are carried forward.
ARG GLM53_RELEASE_VERSION=0.1.0
ARG GLM53_RELEASE_CANDIDATE=18
ARG GLM53_CACHE_SCHEMA=v10
ARG GLM53_SGLANG_BASE=lmsysorg/sglang@sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_BASE_TAG=glm-5.3-flash
ARG GLM53_SGLANG_BASE_INDEX=sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST=sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_REPOSITORY=https://github.com/ormandj/sglang.git
ARG GLM53_SGLANG_HEAD=42a56dc505f775d6f54e9d27a9b57c66023420a0
ARG GLM53_SGLANG_TREE=16eb1fe669e54253b16d206a21e79e9cc7ea6132
ARG GLM53_FLASHINFER_REPOSITORY=https://github.com/ormandj/flashinfer.git
ARG GLM53_FLASHINFER_VERSION=0.6.18
ARG GLM53_FLASHINFER_HEAD=008122fa75c7a27c839feea57a6ef8e8846fa265
ARG GLM53_FLASHINFER_TREE=0462b5c545eb145cb96001285afe4cc89de30605
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
ARG GLM53_SGLANG_TREE
ARG GLM53_FLASHINFER_REPOSITORY
ARG GLM53_FLASHINFER_VERSION
ARG GLM53_FLASHINFER_HEAD
ARG GLM53_FLASHINFER_TREE
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
    FLASHINFER_CUDA_ARCH_LIST=12.0f

# Replace the unverifiable vendor Python tree with the exact integration tree.
# The branch is based on current SGLang main and contains GLM-5.3 support,
# raw-layout FP8 TileLang DSA, and the E4M3-K32 W4A16 loader/runner contract.
RUN set -eux; \
    git init -q "${SGLANG_SOURCE_ROOT}"; \
    cd "${SGLANG_SOURCE_ROOT}"; \
    git remote add origin "${GLM53_SGLANG_REPOSITORY}"; \
    git fetch --depth=1 origin "${GLM53_SGLANG_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${GLM53_SGLANG_HEAD}"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "${GLM53_SGLANG_TREE}"; \
    uv run --no-project --python /opt/sglang/bin/python python -m compileall -q \
      python/sglang/srt/models/glm5_next.py \
      python/sglang/srt/layers/attention/dsa_backend.py \
      python/sglang/srt/layers/quantization/modelopt_quant.py \
      python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py

# Build FlashInfer from the exact SM120 integration tree. It contains
# upstream's >INT32 W4A16 expert-bank correction plus the explicit ModelOpt
# E2M1/E4M3-K32 preparation contract. Generic cubin bundles are deliberately
# removed so this image cannot silently select a kernel built for another GPU.
RUN set -eux; \
    git init -q /tmp/flashinfer-source; \
    cd /tmp/flashinfer-source; \
    git remote add origin "${GLM53_FLASHINFER_REPOSITORY}"; \
    git fetch --depth=1 origin "${GLM53_FLASHINFER_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${GLM53_FLASHINFER_HEAD}"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "${GLM53_FLASHINFER_TREE}"; \
    git submodule update --init --recursive --depth=1; \
    uv pip uninstall --python /opt/sglang/bin/python \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache || true; \
    BUILD_NVEP=0 FLASHINFER_CUDA_ARCH_LIST=12.0f \
      uv pip install --python /opt/sglang/bin/python --no-deps .; \
    cd /; \
    rm -rf /tmp/flashinfer-source

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

# Fail the image build if any part of the intended contract is shadowed by the
# vendor tree or omitted from the installed packages.
RUN set -eux; \
    uv run --no-project --python /opt/sglang/bin/python python -c "\
import importlib.metadata as md; import inspect; import sglang; import flashinfer; \
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration; \
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass; \
from flashinfer.fused_moe.cute_dsl.b12x_moe import b12x_fused_moe; \
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import prepare_w4a16_modelopt_e4m3_k32_weights; \
assert inspect.getfile(sglang).startswith('/opt/sglang-source/python/'); \
assert flashinfer.__version__ == '${GLM53_FLASHINFER_VERSION}', flashinfer.__version__; \
assert flashinfer.__git_commit__ == '${GLM53_FLASHINFER_HEAD}', flashinfer.__git_commit__; \
assert md.version('nvidia-modelopt') == '${GLM53_MODELOPT_VERSION}'; \
print(Glm5NextForConditionalGeneration.__name__, flashinfer.__version__, md.version('nvidia-modelopt'))"

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
      ai.flashinfer.repository=${GLM53_FLASHINFER_REPOSITORY} \
      ai.flashinfer.head=${GLM53_FLASHINFER_HEAD} \
      ai.flashinfer.tree=${GLM53_FLASHINFER_TREE} \
      ai.modelopt.repository=${GLM53_MODELOPT_REPOSITORY} \
      ai.modelopt.release-tag=${GLM53_MODELOPT_RELEASE_TAG} \
      ai.modelopt.head=${GLM53_MODELOPT_HEAD} \
      ai.modelopt.tree=${GLM53_MODELOPT_TREE}
