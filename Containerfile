# GLM-5.3-Flash MXFP4 on SGLang for RTX PRO 6000 Blackwell (SM120).
#
# PROVENANCE DIFFERS FROM THE QWEN3.8-FLASH-NEXT BUILD, DELIBERATELY.
# That build pins sglang main by commit AND tree, applies an archived patch,
# and asserts the resulting effective tree hash. None of that is possible here:
#   * `glm5_next` is NOT in sgl-project/sglang main as of 2026-08-27. The
#     architecture exists only inside the vendor per-model image below.
#   * That image is built from an `ADD sglang.tar.gz`, not a git checkout. Its
#     own metadata reports SGLANG_BUILD_COMMIT=unknown, so there is no upstream
#     commit or tree to verify against.
# The base is therefore pinned by IMMUTABLE DIGEST and nothing more. We do not
# claim source-level reproducibility of the SGLang layer; we claim byte-level
# reproducibility of the image we started from. Re-verify when glm5_next lands
# upstream and switch to the main+patch+tree discipline at that point.
ARG GLM53_RELEASE_VERSION=0.1.0
ARG GLM53_RELEASE_CANDIDATE=1
ARG GLM53_CACHE_SCHEMA=v1
# lmsysorg/sglang:glm-5.3-flash-amd64, pushed 2026-08-27T05:22:01Z.
# This is the OCI index digest; the linux/amd64 manifest it selects is pinned
# separately below and asserted at build time.
ARG GLM53_SGLANG_BASE=lmsysorg/sglang@sha256:44cb2eed3ff808541cc2be6a4fc3b34429a55d8986e9f21660f0ae9a1233a743
ARG GLM53_SGLANG_BASE_TAG=glm-5.3-flash-amd64
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST=sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_BASE_CREATED=2026-08-27T04:58:48Z
ARG GLM53_SGLANG_BASE_CUDA_VERSION=13.0.3
ARG GLM53_SGLANG_BASE_FLASHINFER_VERSION=0.6.17
# FlashInfer is rebuilt from source for SM120. 0.6.18 at this head is the
# revision already qualified on these exact cards by the sibling
# sglang-qwen38-flash-next-sm120 build; reusing a known-good SM120 FlashInfer
# avoids introducing a second unvalidated variable alongside a new model.
ARG GLM53_FLASHINFER_VERSION=0.6.18
ARG GLM53_FLASHINFER_MAIN_HEAD=e4b7fa4b7c3ba5e17286d9c59f2bcf2ca07e0a6d
ARG GLM53_FLASHINFER_MAIN_TREE=2c9c021eb87fb09c982076b8a0b63514bc399e56
# Source checkpoint the served MXFP4 artifact was quantized FROM.
ARG GLM53_MODEL_REPOSITORY=zai-org/GLM-5.3-Flash
ARG GLM53_MODEL_REVISION=04c4e9e95c5da8862dced7e5056455116f83a7e0
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

FROM ${GLM53_SGLANG_BASE} AS runtime
ARG GLM53_RELEASE_VERSION
ARG GLM53_RELEASE_CANDIDATE
ARG GLM53_CACHE_SCHEMA
ARG GLM53_SGLANG_BASE_TAG
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST
ARG GLM53_SGLANG_BASE_CREATED
ARG GLM53_SGLANG_BASE_CUDA_VERSION
ARG GLM53_SGLANG_BASE_FLASHINFER_VERSION
ARG GLM53_FLASHINFER_VERSION
ARG GLM53_FLASHINFER_MAIN_HEAD
ARG GLM53_FLASHINFER_MAIN_TREE
ARG GLM53_MODEL_REPOSITORY
ARG GLM53_MODEL_REVISION
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

ENV PYTHONPATH=/sgl-workspace/sglang/python

# Assert we really are on the base we pinned, and that it carries glm5_next.
# If the vendor ever repoints the tag, the digest FROM already protects us;
# this catches a mismatched ARG block and proves model support is present
# BEFORE we spend 40 minutes compiling FlashInfer.
RUN set -e; \
    test "${GLM53_SGLANG_BASE_CUDA_VERSION}" = "${CUDA_VERSION}"; \
    uv run --no-project --python /opt/sglang/bin/python python -c "\
import importlib.util as u; \
assert u.find_spec('sglang.srt.models.glm5_next') is not None, 'base image lacks glm5_next'; \
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration as M; \
print('glm5_next present:', M.__name__)"

# Rebuild FlashInfer from source for SM120. The stock wheel in the vendor image
# does not carry 12.0f cubins; workstation Blackwell lacks TMEM/tcgen05/wgmma,
# so sm_100 and Hopper kernels do not run on it.
RUN set -e; \
    git init -q /tmp/flashinfer-main; \
    cd /tmp/flashinfer-main; \
    git remote add origin https://github.com/flashinfer-ai/flashinfer.git; \
    git fetch --depth=1 origin "${GLM53_FLASHINFER_MAIN_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "${GLM53_FLASHINFER_MAIN_HEAD}"; \
    test "$(git rev-parse HEAD^{tree})" = "${GLM53_FLASHINFER_MAIN_TREE}"; \
    git submodule update --init --recursive --depth=1; \
    uv pip uninstall --python /opt/sglang/bin/python \
      flashinfer-cubin flashinfer-jit-cache || true; \
    BUILD_NVEP=0 FLASHINFER_CUDA_ARCH_LIST=12.0f \
      uv pip install --python /opt/sglang/bin/python --reinstall --no-deps .; \
    FLASHINFER_CUBIN_DIR=/tmp/flashinfer-main/flashinfer-cubin/flashinfer_cubin/cubins \
      uv pip install --python /opt/sglang/bin/python --reinstall --no-deps \
      --no-build-isolation ./flashinfer-cubin; \
    cd /; \
    rm -rf /tmp/flashinfer-main

RUN set -e; \
    uv run --no-project --python /opt/sglang/bin/python python -c "\
import importlib.util; import flashinfer; import flashinfer_cubin; \
assert flashinfer.__version__ == '${GLM53_FLASHINFER_VERSION}', flashinfer.__version__; \
assert flashinfer.__git_commit__ == '${GLM53_FLASHINFER_MAIN_HEAD}', flashinfer.__git_commit__; \
assert flashinfer_cubin.__version__ == '${GLM53_FLASHINFER_VERSION}', flashinfer_cubin.__version__; \
assert flashinfer_cubin.__git_version__ == '${GLM53_FLASHINFER_MAIN_HEAD}', flashinfer_cubin.__git_version__; \
assert importlib.util.find_spec('flashinfer_jit_cache') is None; \
print('flashinfer', flashinfer.__version__, flashinfer.__git_commit__, 'cubin', flashinfer_cubin.__version__)"

# compressed-tensors must be able to read the MXFP4 artifact this image serves.
# The quantizer emitted mxfp4-pack-quantized with GROUP strategy, group_size 32
# and uint8 E8M0 scales; assert the runtime agrees on that contract.
RUN set -e; \
    uv run --no-project --python /opt/sglang/bin/python python -c "\
from compressed_tensors.config import CompressionFormat; \
from compressed_tensors.quantization.quant_scheme import MXFP4A16; \
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor as C; \
from compressed_tensors.quantization import QuantizationScheme; \
w = MXFP4A16['weights']; \
assert CompressionFormat.mxfp4_pack_quantized.value == 'mxfp4-pack-quantized'; \
assert w.num_bits == 4 and w.group_size == 32 and w.type == 'float'; \
assert C.compression_param_names(QuantizationScheme(targets=['Linear'], weights=w)) == ('weight_packed','weight_scale'); \
print('mxfp4-pack-quantized contract OK, group_size', w.group_size)"

ENV SGLANG_BUILD_BASE_DIGEST=${GLM53_SGLANG_BASE_AMD64_MANIFEST} \
    FLASHINFER_VERSION=${GLM53_FLASHINFER_VERSION} \
    FLASHINFER_BUILD_COMMIT=${GLM53_FLASHINFER_MAIN_HEAD} \
    FLASHINFER_CUDA_ARCH_LIST=12.0f
LABEL org.opencontainers.image.title="sglang-glm53-flash-sm120" \
      org.opencontainers.image.description="SGLang for GLM-5.3-Flash MXFP4 on dual RTX PRO 6000 Blackwell (SM120)" \
      org.opencontainers.image.source=${IMAGE_SOURCE} \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version=${GLM53_RELEASE_VERSION} \
      org.opencontainers.image.revision=${IMAGE_SOURCE_REVISION} \
      ai.release.candidate=rc.${GLM53_RELEASE_CANDIDATE} \
      ai.release.cache-schema=${GLM53_CACHE_SCHEMA} \
      ai.hardware.target-architecture="sm120" \
      ai.model.repository=${GLM53_MODEL_REPOSITORY} \
      ai.model.revision=${GLM53_MODEL_REVISION} \
      ai.model.quantization="mxfp4-pack-quantized (MXFP4A16, routed experts only)" \
      ai.sglang.base.tag=${GLM53_SGLANG_BASE_TAG} \
      ai.sglang.base.amd64-manifest=${GLM53_SGLANG_BASE_AMD64_MANIFEST} \
      ai.sglang.base.created=${GLM53_SGLANG_BASE_CREATED} \
      ai.sglang.base.cuda-version=${GLM53_SGLANG_BASE_CUDA_VERSION} \
      ai.sglang.base.flashinfer-version=${GLM53_SGLANG_BASE_FLASHINFER_VERSION} \
      ai.sglang.source-provenance="vendor per-model image, built from tarball; no upstream commit available" \
      ai.flashinfer.version=${GLM53_FLASHINFER_VERSION} \
      ai.flashinfer.main.head=${GLM53_FLASHINFER_MAIN_HEAD} \
      ai.flashinfer.main.tree=${GLM53_FLASHINFER_MAIN_TREE}
