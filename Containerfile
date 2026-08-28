# GLM-5.3-Flash MXFP4 on SGLang for RTX PRO 6000 Blackwell (SM120).
#
# PROVENANCE DIFFERS FROM THE QWEN3.8-FLASH-NEXT BUILD, DELIBERATELY.
# That build pins sglang main by commit AND tree, applies an archived patch,
# and asserts the resulting effective tree hash. None of that is possible here:
#   * `glm5_next` is NOT in sgl-project/sglang main as of 2026-08-28. The
#     architecture exists only inside the vendor per-model image below.
#   * That image is built from an `ADD sglang.tar.gz`, not a git checkout. Its
#     own metadata reports SGLANG_BUILD_COMMIT=unknown, so there is no upstream
#     commit or tree to verify against.
# The base is therefore pinned by IMMUTABLE DIGEST and nothing more. We do not
# claim source-level reproducibility of the SGLang layer; we claim byte-level
# reproducibility of the image we started from. Re-verify when glm5_next lands
# upstream and switch to the main+patch+tree discipline at that point.
ARG GLM53_RELEASE_VERSION=0.1.0
ARG GLM53_RELEASE_CANDIDATE=6
ARG GLM53_CACHE_SCHEMA=v5
# lmsysorg/sglang:glm-5.3-flash-amd64, pushed 2026-08-27T05:22:01Z.
# This is the OCI index digest; the linux/amd64 manifest it selects is pinned
# separately below and asserted at build time.
ARG GLM53_SGLANG_BASE=lmsysorg/sglang@sha256:44cb2eed3ff808541cc2be6a4fc3b34429a55d8986e9f21660f0ae9a1233a743
ARG GLM53_SGLANG_BASE_TAG=glm-5.3-flash-amd64
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST=sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_BASE_CREATED=2026-08-27T04:58:48Z
ARG GLM53_SGLANG_BASE_CUDA_VERSION=13.0.3
ARG GLM53_SGLANG_BASE_FLASHINFER_VERSION=0.6.17
# FlashInfer is rebuilt from the exact main head refreshed on 2026-08-28.
ARG GLM53_FLASHINFER_VERSION=0.6.18
ARG GLM53_FLASHINFER_MAIN_HEAD=71d31b5a23a3c0394edb36330dec1ce2a0def365
ARG GLM53_FLASHINFER_MAIN_TREE=a2577ad013205dfc996f6ffe5259f3102a2cd075
# Source checkpoint the served MXFP4 artifact was quantized FROM.
ARG GLM53_MODEL_REPOSITORY=zai-org/GLM-5.3-Flash-BF16
ARG GLM53_MODEL_REVISION=f12e0fe1f6b2ea274c11a569582edfd99d993c5e
# The vendor image lacks git provenance. These byte-level pre/postimage pins
# make our three source modifications fail closed without implying a git commit.
ARG GLM53_SGLANG_MXFP4_PREIMAGE_SHA256=38fe76f6a3c3dd142feea2a0e9ad685962cf6a4b8bf709f2d49b765840884dcb
ARG GLM53_SGLANG_MXFP4_POSTIMAGE_SHA256=4a5fdcfca8edb681e8b2e781e9cddf9545c866301b5ce2d16aa1545061791f09
ARG GLM53_SGLANG_GLM5_NEXT_PREIMAGE_SHA256=0a141565e73252ddb7f1773f30f0c48e001b7dce21a5ca7864b4ea6ae51d0ccd
ARG GLM53_SGLANG_GLM5_NEXT_POSTIMAGE_SHA256=ed1021e7fd3d9d31f5b97979e8dc12f158cd5ea9bda1d9d42b017c2305953274
ARG GLM53_SGLANG_FLASH_MLA_SM120_PREIMAGE_SHA256=39f0f98151a7cfd750b987d82cf05fafe80e8e972ef53a2b78352ce9b472e9b5
ARG GLM53_SGLANG_FLASH_MLA_SM120_POSTIMAGE_SHA256=052bd1ca3f63b2fd569aad3b55bf0f3d07d157773b5d2ddeae981ac742755e93
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
ARG GLM53_SGLANG_MXFP4_PREIMAGE_SHA256
ARG GLM53_SGLANG_MXFP4_POSTIMAGE_SHA256
ARG GLM53_SGLANG_GLM5_NEXT_PREIMAGE_SHA256
ARG GLM53_SGLANG_GLM5_NEXT_POSTIMAGE_SHA256
ARG GLM53_SGLANG_FLASH_MLA_SM120_PREIMAGE_SHA256
ARG GLM53_SGLANG_FLASH_MLA_SM120_POSTIMAGE_SHA256
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

COPY patches/0001-mxfp4-sm120-preserve-non-gpt-oss-moe-semantics.patch /usr/share/sglang-glm53-flash-sm120/patches/0001-mxfp4-sm120-preserve-non-gpt-oss-moe-semantics.patch
COPY patches/test_glm53_sm120_mxfp4_patch.py /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_sm120_mxfp4_patch.py
COPY patches/0002-glm53-dflash-hidden-state-capture.patch /usr/share/sglang-glm53-flash-sm120/patches/0002-glm53-dflash-hidden-state-capture.patch
COPY patches/0003-glm53-dflash-mhc-residual-none.patch /usr/share/sglang-glm53-flash-sm120/patches/0003-glm53-dflash-mhc-residual-none.patch
COPY patches/test_glm53_dflash_patch.py /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_dflash_patch.py
COPY patches/test_glm53_mixed_precision_contract.py /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_mixed_precision_contract.py
COPY patches/0004-glm53-sm120-sparse-mla-architecture.patch /usr/share/sglang-glm53-flash-sm120/patches/0004-glm53-sm120-sparse-mla-architecture.patch
COPY patches/test_glm53_sm120_sparse_mla_patch.py /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_sm120_sparse_mla_patch.py

# Patch only exact vendor bytes. The first fix preserves GLM's contiguous
# gate/up layout and standard clamped-SwiGLU semantics in the SM120 MXFP4 path.
# The second and third are the upstream DFlash2 hidden-state capture changes
# from #36708 and its mHC residual=None correction from #36755. The fourth
# extends upstream #26928's fail-closed SM120 sparse-MLA architecture allowlist
# to the native GLM-5.3 target and NextN class names.
RUN set -e; \
    cd /sgl-workspace/sglang; \
    test "$(sha256sum python/sglang/srt/layers/quantization/mxfp4.py | cut -d' ' -f1)" = "${GLM53_SGLANG_MXFP4_PREIMAGE_SHA256}"; \
    test "$(sha256sum python/sglang/srt/models/glm5_next.py | cut -d' ' -f1)" = "${GLM53_SGLANG_GLM5_NEXT_PREIMAGE_SHA256}"; \
    test "$(sha256sum python/sglang/kernels/ops/attention/flash_mla_sm120.py | cut -d' ' -f1)" = "${GLM53_SGLANG_FLASH_MLA_SM120_PREIMAGE_SHA256}"; \
    patch --fuzz=0 -p1 -i /usr/share/sglang-glm53-flash-sm120/patches/0001-mxfp4-sm120-preserve-non-gpt-oss-moe-semantics.patch; \
    patch --fuzz=0 -p1 -i /usr/share/sglang-glm53-flash-sm120/patches/0002-glm53-dflash-hidden-state-capture.patch; \
    patch --fuzz=0 -p1 -i /usr/share/sglang-glm53-flash-sm120/patches/0003-glm53-dflash-mhc-residual-none.patch; \
    patch --fuzz=0 -p1 -i /usr/share/sglang-glm53-flash-sm120/patches/0004-glm53-sm120-sparse-mla-architecture.patch; \
    test "$(sha256sum python/sglang/srt/layers/quantization/mxfp4.py | cut -d' ' -f1)" = "${GLM53_SGLANG_MXFP4_POSTIMAGE_SHA256}"; \
    test "$(sha256sum python/sglang/srt/models/glm5_next.py | cut -d' ' -f1)" = "${GLM53_SGLANG_GLM5_NEXT_POSTIMAGE_SHA256}"; \
    test "$(sha256sum python/sglang/kernels/ops/attention/flash_mla_sm120.py | cut -d' ' -f1)" = "${GLM53_SGLANG_FLASH_MLA_SM120_POSTIMAGE_SHA256}"; \
    uv run --no-project --python /opt/sglang/bin/python python \
      /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_sm120_mxfp4_patch.py; \
    uv run --no-project --python /opt/sglang/bin/python python \
      /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_dflash_patch.py; \
    uv run --no-project --python /opt/sglang/bin/python python \
      /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_mixed_precision_contract.py; \
    uv run --no-project --python /opt/sglang/bin/python python \
      /usr/share/sglang-glm53-flash-sm120/tests/test_glm53_sm120_sparse_mla_patch.py

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

# The GLM support branch enters the shared DeepSeek/DSA hook but misses the
# SM120 fallback block currently applied only to DeepseekV4ForCausalLM. On
# workstation Blackwell, leaving those defaults enabled reaches DeepGEMM's
# unavailable tcgen05/TMEM mHC kernel and crashes during warmup. Mirror the
# upstream DeepSeek-V4 SM120 settings in the hardware-specific image until the
# architecture hook is generalized upstream.
ENV SGLANG_OPT_FP8_WO_A_GEMM=0 \
    SGLANG_OPT_USE_TOPK_V2=0 \
    SGLANG_OPT_USE_TILELANG_MHC_PRE=0 \
    SGLANG_OPT_FUSE_MHC_POST_PRE=1 \
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
    SGLANG_OPT_USE_TILELANG_INDEXER=1

RUN set -e; \
    uv run --no-project --python /opt/sglang/bin/python python -c "\
from sglang.srt.environ import envs; \
assert not envs.SGLANG_OPT_FP8_WO_A_GEMM.get(); \
assert not envs.SGLANG_OPT_USE_TOPK_V2.get(); \
assert not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get(); \
assert envs.SGLANG_OPT_FUSE_MHC_POST_PRE.get(); \
assert not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get(); \
assert envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get(); \
assert envs.SGLANG_OPT_USE_TILELANG_INDEXER.get(); \
print('GLM SM120 DeepGEMM/mHC/indexer fallback contract OK')"

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
