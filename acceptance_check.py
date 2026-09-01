import importlib.metadata as md
import inspect
import os
import sglang
import flashinfer
from types import SimpleNamespace
from sglang.srt.arg_groups import model_hook
from sglang.srt.environ import envs
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.srt.models.deepseek_nextn import DeepseekModelNextN
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.kernels.ops.attention.fla import chunk_intra, kda
from sglang.kernels.ops.attention.fla import fused_kda_conv_recurrent_verify
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass
from sglang.srt.layers.quantization import modelopt_quant
from sglang.kernels.ops.attention import flash_mla_sm120
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool, kpool_fp8_index
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache import unified_radix_cache
from sglang.srt.mem_cache.unified_cache.components import full_component
from sglang.srt.utils import async_probe
from flashinfer.mla import SparseMLASm120Wrapper
from flashinfer.mla._sparse_mla_sm120 import _bytes_per_token_for_model_type, _MODEL_TYPE_GLM53_NOPE
from flashinfer.fused_moe.cute_dsl.b12x_moe import b12x_fused_moe
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import prepare_w4a16_modelopt_e4m3_k32_weights
assert inspect.getfile(sglang).startswith('/opt/sglang-source/python/')
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:2:16:8'
assert flashinfer.__version__ == os.environ['GLM53_FLASHINFER_VERSION'], flashinfer.__version__
assert flashinfer.__git_commit__ == os.environ['GLM53_FLASHINFER_HEAD'], flashinfer.__git_commit__
assert md.version('nvidia-modelopt') == os.environ['GLM53_MODELOPT_VERSION']
Q=type('Q',(),{'get_name':lambda self:'modelopt_fp4'})
q=Q()
assert DeepseekModelNextN._resolve_modelopt_fp4_quant_config(q,False) is None
assert DeepseekModelNextN._resolve_modelopt_fp4_quant_config(q,True) is q
g=Glm5NextForConditionalGenerationNextN.__new__(Glm5NextForConditionalGenerationNextN)
assert g._resolve_nextn_quant_config(SimpleNamespace(num_hidden_layers=45,quantization_config={'ignore':['*.self_attn.*']}),q) is q
assert g._resolve_nextn_quant_config(SimpleNamespace(num_hidden_layers=45,quantization_config={'ignore':['model.layers.45.*']}),q) is None
eagle_init=inspect.getsource(EagleDraftWorker.__init__)
assert 'self._init_dsa_index_share_state()' in eagle_init
assert 'self.init_lm_head()' not in eagle_init
eagle_pool_init=inspect.getsource(EagleDraftWorker.alloc_memory_pool)
assert eagle_pool_init.index('self.draft_worker.alloc_memory_pool') < eagle_pool_init.index('self.init_lm_head()')
assert '_embed_and_head_shared' not in inspect.getsource(EagleDraftWorker.init_lm_head)
model_hook.is_sm120_supported=lambda:True
model_hook._apply_glm5_next_sm120_defaults('Glm5NextForConditionalGeneration')
assert envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get() is False
assert inspect.getsource(mhc.mhc_fused_post_pre).count('_retain_mhc_capture_owners(') == 4
assert inspect.getsource(mhc.mhc_pre).count('_retain_mhc_capture_owners(') == 4
assert '@torch.compiler.disable' in inspect.getsource(mhc._retain_mhc_capture_owners)
assert 'reuse_input_storage=True' in inspect.getsource(modelopt_quant._prepare_flashinfer_b12x_w4a16_inplace)
assert '_prepared_weights=quant_info.prepared_weights' in inspect.getsource(flashinfer_cutlass._run_flashinfer_b12x_w4a16)
assert callable(kda.precompile_kda_prefill_kernels)
assert not hasattr(dsa_indexer_kpool, '_get_compress_gate_stream')
assert 'torch.cuda.Stream()' in inspect.getsource(dsa_indexer_kpool.IndexerKPool.__init__)
assert callable(dsa_indexer_kpool.IndexerKPool._get_logits_head_gate_compiled)
gate_owner_source=inspect.getsource(dsa_indexer_kpool.IndexerKPool._get_logits_head_gate)
assert 'return self._retain_logits_head_gate_capture_owner(weights)' in gate_owner_source
gate_retain_source=inspect.getsource(dsa_indexer_kpool.IndexerKPool._retain_logits_head_gate_capture_owner)
assert '@torch.compiler.disable' in gate_retain_source
assert 'retain_full_cuda_graph_owner(weights)' in gate_retain_source
assert 'is_current_stream_capturing()' not in gate_retain_source
paged_logits_owner_source=inspect.getsource(dsa_indexer_kpool.IndexerKPool._retain_paged_logits_capture_owner)
assert '@torch.compiler.disable' in paged_logits_owner_source
assert 'retain_full_cuda_graph_owner(logits)' in paged_logits_owner_source
assert 'register_graph_buffer(logits' in paged_logits_owner_source
paged_topk_source=inspect.getsource(dsa_indexer_kpool.IndexerKPool._get_topk_paged)
assert paged_topk_source.index('_retain_paged_logits_capture_owner(logits, layer_id)') < paged_topk_source.index('topk_result = self._topk_from_kpool_logits(')
assert callable(kpool_fp8_index.precompile_index_prefix_gather)
assert callable(flashinfer_cutlass.precompile_w4a16_prefill_routes)
assert '(256, 320, 512, 1024, 2048, 4096, 8192)' in inspect.getsource(Glm5NextForConditionalGeneration.precompile_kernels_after_loading)
assert 'CACHE_RING' in inspect.getsource(fused_kda_conv_recurrent_verify)
assert '_replayssm_ring_ok' in inspect.getsource(KDAAttnBackend._can_run_fused_chain_verify)
assert '_supports_linear_replayssm_spec' in inspect.getsource(kv_cache_configurator.KVCacheConfigurator)
assert 'mamba2_cache_params.is_kda' in inspect.getsource(kv_cache_configurator.KVCacheConfigurator._supports_linear_replayssm_spec)
assert 'GLM' not in inspect.getsource(chunk_intra._get_kda_intra_static_config)
assert 'precompile_kda_prefill_kernels' in inspect.getsource(Glm5NextForConditionalGeneration.precompile_kernels_after_loading)
assert 'Glm5NextForConditionalGeneration' in flash_mla_sm120._GLM_DSA_MODEL_ARCHS
assert flash_mla_sm120._GLM53_NOPE_FLASHINFER_TOPK == 2176
assert flash_mla_sm120._GLM53_NOPE_FLASHINFER_KV_DIM == 656
assert 'q.shape[-1] == 512' in inspect.getsource(flash_mla_sm120.flashinfer_sparse_mla_forward)
assert 'qk_nope_head_dim == 512' not in inspect.getsource(flash_mla_sm120.flashinfer_sparse_mla_forward)
assert 'if uses_flashinfer_sparse_mla and is_glm_sm12_fp8:' in inspect.getsource(flash_mla_sm120._validate_flashinfer_sparse_mla_backend)
assert 'return 656' in inspect.getsource(kv_cache_configurator.calculate_mla_kv_cache_dim)
alloc_extend_source=inspect.getsource(PagedTokenToKVPoolAllocator.alloc_extend)
assert 'alloc_extend last_loc' in alloc_extend_source
assert 'alloc_extend free_pages' in alloc_extend_source
assert 'alloc_extend output' in alloc_extend_source
assert 'index == 0 (reserved / unwritten slot?)' in inspect.getsource(async_probe.maybe_detect_oob)
assert 'invalid_count=' in inspect.getsource(async_probe.maybe_sync_detect_oob)
assert callable(async_probe.maybe_sync_detect_oob)
assert 'positions=' in inspect.getsource(async_probe.maybe_sync_detect_oob)
assert 'UnifiedRadixCache.cache_finished_req insert values' in inspect.getsource(unified_radix_cache.UnifiedRadixCache.cache_finished_req)
assert 'UnifiedRadixCache.cache_unfinished_req insert values' in inspect.getsource(unified_radix_cache.UnifiedRadixCache.cache_unfinished_req)
assert 'allocator free aliases reachable Full value' in inspect.getsource(unified_radix_cache.UnifiedRadixCache._debug_assert_frees_not_reachable)
assert 'reachable Full value changed' in inspect.getsource(unified_radix_cache.UnifiedRadixCache._debug_assert_full_value_snapshot_unchanged)
assert 'FullComponent.redistribute_on_node_split source' in inspect.getsource(full_component.FullComponent.redistribute_on_node_split)
assert 'FullComponent.evict_component value' in inspect.getsource(full_component.FullComponent.evict_component)
assert _bytes_per_token_for_model_type(_MODEL_TYPE_GLM53_NOPE) == 656
assert callable(getattr(SparseMLASm120Wrapper, 'run', None))
print(Glm5NextForConditionalGeneration.__name__, flashinfer.__version__, md.version('nvidia-modelopt'))
