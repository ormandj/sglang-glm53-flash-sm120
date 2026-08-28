"""Build-time contract for the GLM-5.3 SM120 sparse-MLA integration."""

import sys
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _GLM_DSA_MODEL_ARCHS,
    _validate_flashinfer_sparse_mla_backend,
    flashinfer_sparse_mla_forward,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend


SPARSE = "flashinfer_sparse_mla"
TARGET = "Glm5NextForConditionalGeneration"
NEXTN = "Glm5NextForConditionalGenerationNextN"


def validate(arch: str, *, sm: int = 12, dtype=torch.float8_e4m3fn) -> bool:
    return _validate_flashinfer_sparse_mla_backend(
        model_arch=arch,
        device_sm_major=sm,
        kv_cache_dtype=dtype,
        prefill_impl=SPARSE,
        decode_impl=SPARSE,
    )


def must_reject(**kwargs) -> None:
    try:
        _validate_flashinfer_sparse_mla_backend(**kwargs)
    except ValueError:
        return
    raise AssertionError(f"sparse-MLA contract unexpectedly accepted {kwargs!r}")


assert TARGET in _GLM_DSA_MODEL_ARCHS
assert NEXTN in _GLM_DSA_MODEL_ARCHS
assert validate(TARGET)
assert validate(NEXTN)

# Preserve the original architectures while keeping every non-architecture
# condition fail closed.
assert validate("GlmMoeDsaForCausalLM")
assert validate("GlmMoeDsaForCausalLMNextN")

common = {
    "model_arch": TARGET,
    "device_sm_major": 12,
    "kv_cache_dtype": torch.float8_e4m3fn,
    "prefill_impl": SPARSE,
    "decode_impl": SPARSE,
}
must_reject(**(common | {"model_arch": "UnrelatedForCausalLM"}))
must_reject(**(common | {"device_sm_major": 10}))
must_reject(**(common | {"kv_cache_dtype": torch.bfloat16}))
must_reject(**(common | {"decode_impl": "trtllm"}))


# The 584-byte GLM-5.3/DSv4 cache must use FlashInfer's supported dual-segment
# ABI: 128 primary candidates plus the remaining 1,923 base/tail candidates,
# both reading the same physical cache. Full-width length tensors deliberately
# preserve live KPool tail entries after any -1 padding in the base table.
captured_dsv4 = {}
captured_generic = {}


def fake_dsv4(**kwargs):
    captured_dsv4.update(kwargs)
    query = kwargs["query"]
    return query.new_full((*query.shape[:-1], 512), 3)


def fake_generic(**kwargs):
    captured_generic.update(kwargs)
    query = kwargs["query"]
    return query.new_full((*query.shape[:-1], kwargs["kv_lora_rank"]), 5)


flashinfer = ModuleType("flashinfer")
flashinfer.__path__ = []
mla = ModuleType("flashinfer.mla")
mla.trtllm_batch_decode_sparse_mla_dsv4 = fake_dsv4
mla.trtllm_batch_decode_with_kv_cache_mla = fake_generic
flashinfer.mla = mla

indices = torch.arange(2051, dtype=torch.int32).view(1, -1).repeat(2, 1)
indices[:, 1536:2048] = -1
with patch.dict(sys.modules, {"flashinfer": flashinfer, "flashinfer.mla": mla}):
    output = flashinfer_sparse_mla_forward(
        q=torch.zeros((2, 16, 512), dtype=torch.bfloat16),
        kv_cache=torch.zeros((2, 64, 1, 584), dtype=torch.uint8),
        indices=indices,
        seq_lens=torch.tensor([1536, 1536], dtype=torch.int32),
        workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
        page_size=64,
        kv_cache_dim=584,
        qk_nope_head_dim=256,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        sm_scale=0.125,
        skip_softmax_threshold_scale_factor=None,
    )

assert tuple(captured_dsv4["query"].shape) == (2, 1, 16, 512)
assert tuple(captured_dsv4["sparse_indices"].shape) == (2, 128)
assert tuple(captured_dsv4["extra_sparse_indices"].shape) == (2, 1923)
assert captured_dsv4["extra_sparse_indices"][:, -3:].tolist() == [
    [2048, 2049, 2050],
    [2048, 2049, 2050],
]
assert captured_dsv4["swa_topk_lens"].tolist() == [128, 128]
assert captured_dsv4["extra_sparse_topk_lens"].tolist() == [1923, 1923]
assert captured_dsv4["swa_kv_cache"].data_ptr() == captured_dsv4[
    "compressed_kv_cache"
].data_ptr()
assert captured_dsv4["kv_layout"] == "NHD"
assert captured_dsv4["backend"] == "sparse"
assert tuple(output.shape) == (2, 16, 512)
assert torch.all(output == 3)

# Preserve the original 656-byte GLM-NSA adapter unchanged.
with patch.dict(sys.modules, {"flashinfer": flashinfer, "flashinfer.mla": mla}):
    generic_output = flashinfer_sparse_mla_forward(
        q=torch.zeros((2, 8, 576), dtype=torch.bfloat16),
        kv_cache=torch.zeros((2, 64, 1, 656), dtype=torch.uint8),
        indices=torch.tensor([[7, 9, -1, -1], [4, 6, 8, -1]], dtype=torch.int32),
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
        page_size=64,
        kv_cache_dim=656,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        sm_scale=0.125,
        skip_softmax_threshold_scale_factor=0.25,
    )

assert captured_generic["sparse_mla_top_k"] == 4
assert captured_generic["kv_scale_format"] == "arbitrary_fp32"
assert tuple(generic_output.shape) == (2, 8, 512)
assert torch.all(generic_output == 5)

# The backend guard opens KPool tails only for the corrected 584-byte path.
backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
backend.dsa_index_kpool = 4
backend.kv_cache_dim = 584
backend._check_kpool_tail_backend(indices, SPARSE, "decode")
backend.kv_cache_dim = 656
try:
    backend._check_kpool_tail_backend(indices, SPARSE, "decode")
except NotImplementedError:
    pass
else:
    raise AssertionError("656-byte sparse-MLA path unexpectedly admitted KPool tails")

print("GLM-5.3 SM120 sparse-MLA architecture and KPool contracts OK")
