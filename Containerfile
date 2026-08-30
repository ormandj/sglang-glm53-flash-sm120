# GLM-5.3-Flash on two RTX PRO 6000 Blackwell GPUs (SM120).
#
# The vendor image supplies the CUDA/PyTorch dependency stack only. Its
# SGLang tree has no git provenance, so this candidate puts an exact, verifiable
# SGLang integration tree first on PYTHONPATH and rebuilds FlashInfer from exact
# official bases plus checksummed project patches. No rc.14 vendor-byte patches
# are carried forward.
ARG GLM53_RELEASE_VERSION=0.1.0
ARG GLM53_RELEASE_CANDIDATE=50
ARG GLM53_CACHE_SCHEMA=v37
ARG GLM53_SGLANG_BASE=lmsysorg/sglang@sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_BASE_TAG=glm-5.3-flash
ARG GLM53_SGLANG_BASE_INDEX=sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6
ARG GLM53_SGLANG_BASE_AMD64_MANIFEST=sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
ARG GLM53_SGLANG_REPOSITORY=https://github.com/sgl-project/sglang.git
ARG GLM53_SGLANG_HEAD=cdbfe90b4a6c728e03e6520862d792501b3a97bb
ARG GLM53_SGLANG_UPSTREAM_TREE=68a9d2477cf06c8e0a737997439272ebdc2da1c8
ARG GLM53_SGLANG_TREE=622f994b2cde2c2bf2ca82d1611a51b15cfd3bbc
ARG GLM53_SGLANG_PATCH_SHA256=74aa261113931778f5eb4dc1fddee424dcfa064655b073d251a3506c523f9f44
ARG GLM53_FLASHINFER_REPOSITORY=https://github.com/flashinfer-ai/flashinfer.git
ARG GLM53_FLASHINFER_VERSION=0.6.18
ARG GLM53_FLASHINFER_HEAD=e425c7b029ca90d5d01ff207913b070863d35a5b
ARG GLM53_FLASHINFER_UPSTREAM_TREE=cd9bf5311aa0521ce3972c2d93481a13836d5268
ARG GLM53_FLASHINFER_TREE=6a957df7b48adac53ac27d2156b46bc2455ce157
ARG GLM53_FLASHINFER_PATCH_SHA256=d57993d3fb2a90672c2a28a8f058194146ebbdafd5979df72995a7cc2ec506be
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

# Fail the image build if any part of the intended contract is shadowed by the
# vendor tree or omitted from the installed packages.
RUN set -eux; \
    uv run --no-project --python /opt/sglang/bin/python python -c "\
import importlib.metadata as md; import inspect; import os; import sglang; import flashinfer; \
from types import SimpleNamespace; \
from sglang.srt.arg_groups import model_hook; \
from sglang.srt.environ import envs; \
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration; \
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN; \
from sglang.srt.models.deepseek_nextn import DeepseekModelNextN; \
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker; \
from sglang.kernels.ops.attention.fla import chunk_intra, kda; \
from sglang.kernels.ops.attention.fla import fused_kda_conv_recurrent_verify; \
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend; \
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass; \
from sglang.srt.layers.quantization import modelopt_quant; \
from sglang.kernels.ops.attention import flash_mla_sm120; \
from sglang.kernels.ops.layernorm import mhc; \
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool, kpool_fp8_index; \
from sglang.srt.mem_cache import kv_cache_configurator; \
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator; \
from sglang.srt.mem_cache import unified_radix_cache; \
from sglang.srt.mem_cache.unified_cache.components import full_component; \
from sglang.srt.utils import async_probe; \
from flashinfer.mla import SparseMLASm120Wrapper; \
from flashinfer.mla._sparse_mla_sm120 import _bytes_per_token_for_model_type, _MODEL_TYPE_GLM53_NOPE; \
from flashinfer.fused_moe.cute_dsl.b12x_moe import b12x_fused_moe; \
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import prepare_w4a16_modelopt_e4m3_k32_weights; \
assert inspect.getfile(sglang).startswith('/opt/sglang-source/python/'); \
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:2:16:8'; \
assert flashinfer.__version__ == '${GLM53_FLASHINFER_VERSION}', flashinfer.__version__; \
assert flashinfer.__git_commit__ == '${GLM53_FLASHINFER_HEAD}', flashinfer.__git_commit__; \
assert md.version('nvidia-modelopt') == '${GLM53_MODELOPT_VERSION}'; \
Q=type('Q',(),{'get_name':lambda self:'modelopt_fp4'}); q=Q(); \
assert DeepseekModelNextN._resolve_modelopt_fp4_quant_config(q,False) is None; \
assert DeepseekModelNextN._resolve_modelopt_fp4_quant_config(q,True) is q; \
g=Glm5NextForConditionalGenerationNextN.__new__(Glm5NextForConditionalGenerationNextN); \
assert g._resolve_nextn_quant_config(SimpleNamespace(num_hidden_layers=45,quantization_config={'ignore':['*.self_attn.*']}),q) is q; \
assert g._resolve_nextn_quant_config(SimpleNamespace(num_hidden_layers=45,quantization_config={'ignore':['model.layers.45.*']}),q) is None; \
eagle_init=inspect.getsource(EagleDraftWorker.__init__); \
assert 'self._init_dsa_index_share_state()' in eagle_init; \
assert 'self.init_lm_head()' not in eagle_init; \
eagle_pool_init=inspect.getsource(EagleDraftWorker.alloc_memory_pool); \
assert eagle_pool_init.index('self.draft_worker.alloc_memory_pool') < eagle_pool_init.index('self.init_lm_head()'); \
assert '_embed_and_head_shared' not in inspect.getsource(EagleDraftWorker.init_lm_head); \
model_hook.is_sm120_supported=lambda:True; model_hook._apply_glm5_next_sm120_defaults('Glm5NextForConditionalGeneration'); \
assert envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get() is False; \
assert inspect.getsource(mhc.mhc_fused_post_pre).count('retain_full_cuda_graph_owner(') == 2; \
assert 'reuse_input_storage=True' in inspect.getsource(modelopt_quant._prepare_flashinfer_b12x_w4a16_inplace); \
assert '_prepared_weights=quant_info.prepared_weights' in inspect.getsource(flashinfer_cutlass._run_flashinfer_b12x_w4a16); \
assert callable(kda.precompile_kda_prefill_kernels); \
assert not hasattr(dsa_indexer_kpool, '_get_compress_gate_stream'); \
assert 'torch.cuda.Stream()' in inspect.getsource(dsa_indexer_kpool.IndexerKPool.__init__); \
assert callable(dsa_indexer_kpool.IndexerKPool._get_logits_head_gate_compiled); \
gate_owner_source=inspect.getsource(dsa_indexer_kpool.IndexerKPool._get_logits_head_gate); \
assert 'return self._retain_logits_head_gate_capture_owner(weights)' in gate_owner_source; \
gate_retain_source=inspect.getsource(dsa_indexer_kpool.IndexerKPool._retain_logits_head_gate_capture_owner); \
assert '@torch.compiler.disable' in gate_retain_source; \
assert 'retain_full_cuda_graph_owner(weights)' in gate_retain_source; \
assert 'is_current_stream_capturing()' not in gate_retain_source; \
assert callable(kpool_fp8_index.precompile_index_prefix_gather); \
assert callable(flashinfer_cutlass.precompile_w4a16_prefill_routes); \
assert '(256, 320, 512, 1024, 2048, 4096, 8192)' in inspect.getsource(Glm5NextForConditionalGeneration.precompile_kernels_after_loading); \
assert 'CACHE_RING' in inspect.getsource(fused_kda_conv_recurrent_verify); \
assert '_replayssm_ring_ok' in inspect.getsource(KDAAttnBackend._can_run_fused_chain_verify); \
assert '_supports_linear_replayssm_spec' in inspect.getsource(kv_cache_configurator.KVCacheConfigurator); \
assert 'mamba2_cache_params.is_kda' in inspect.getsource(kv_cache_configurator.KVCacheConfigurator._supports_linear_replayssm_spec); \
assert 'GLM' not in inspect.getsource(chunk_intra._get_kda_intra_static_config); \
assert 'precompile_kda_prefill_kernels' in inspect.getsource(Glm5NextForConditionalGeneration.precompile_kernels_after_loading); \
assert 'Glm5NextForConditionalGeneration' in flash_mla_sm120._GLM_DSA_MODEL_ARCHS; \
assert flash_mla_sm120._GLM53_NOPE_FLASHINFER_TOPK == 2176; \
assert flash_mla_sm120._GLM53_NOPE_FLASHINFER_KV_DIM == 528; \
assert 'q.shape[-1] == 512' in inspect.getsource(flash_mla_sm120.flashinfer_sparse_mla_forward); \
assert 'qk_nope_head_dim == 512' not in inspect.getsource(flash_mla_sm120.flashinfer_sparse_mla_forward); \
assert 'if uses_flashinfer_sparse_mla and is_glm_sm12_fp8:' in inspect.getsource(flash_mla_sm120._validate_flashinfer_sparse_mla_backend); \
assert 'return 528' in inspect.getsource(kv_cache_configurator.calculate_mla_kv_cache_dim); \
alloc_extend_source=inspect.getsource(PagedTokenToKVPoolAllocator.alloc_extend); \
assert 'alloc_extend last_loc' in alloc_extend_source; \
assert 'alloc_extend free_pages' in alloc_extend_source; \
assert 'alloc_extend output' in alloc_extend_source; \
assert 'index == 0 (reserved / unwritten slot?)' in inspect.getsource(async_probe.maybe_detect_oob); \
assert 'invalid_count=' in inspect.getsource(async_probe.maybe_sync_detect_oob); \
assert callable(async_probe.maybe_sync_detect_oob); \
assert 'positions=' in inspect.getsource(async_probe.maybe_sync_detect_oob); \
assert 'UnifiedRadixCache.cache_finished_req insert values' in inspect.getsource(unified_radix_cache.UnifiedRadixCache.cache_finished_req); \
assert 'UnifiedRadixCache.cache_unfinished_req insert values' in inspect.getsource(unified_radix_cache.UnifiedRadixCache.cache_unfinished_req); \
assert 'UnifiedRadixCache.insert completed tree' in inspect.getsource(unified_radix_cache.UnifiedRadixCache.insert); \
assert 'UnifiedRadixCache.match_prefix pre-walk tree' in inspect.getsource(unified_radix_cache.UnifiedRadixCache.match_prefix); \
assert 'allocator free aliases reachable Full value' in inspect.getsource(unified_radix_cache.UnifiedRadixCache._debug_assert_frees_not_reachable); \
assert 'reachable Full value changed' in inspect.getsource(unified_radix_cache.UnifiedRadixCache._debug_assert_full_value_snapshot_unchanged); \
assert 'FullComponent.redistribute_on_node_split source' in inspect.getsource(full_component.FullComponent.redistribute_on_node_split); \
assert 'FullComponent.evict_component value' in inspect.getsource(full_component.FullComponent.evict_component); \
assert _bytes_per_token_for_model_type(_MODEL_TYPE_GLM53_NOPE) == 528; \
assert callable(getattr(SparseMLASm120Wrapper, 'run', None)); \
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
      ai.sglang.patch-sha256=${GLM53_SGLANG_PATCH_SHA256} \
      ai.flashinfer.repository=${GLM53_FLASHINFER_REPOSITORY} \
      ai.flashinfer.head=${GLM53_FLASHINFER_HEAD} \
      ai.flashinfer.tree=${GLM53_FLASHINFER_TREE} \
      ai.flashinfer.patch-sha256=${GLM53_FLASHINFER_PATCH_SHA256} \
      ai.modelopt.repository=${GLM53_MODELOPT_REPOSITORY} \
      ai.modelopt.release-tag=${GLM53_MODELOPT_RELEASE_TAG} \
      ai.modelopt.head=${GLM53_MODELOPT_HEAD} \
      ai.modelopt.tree=${GLM53_MODELOPT_TREE}
