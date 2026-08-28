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


# The exact GLM-5.3 cache is 528 bytes: 512 E4M3 latent bytes plus four
# arbitrary FP32 scales and no RoPE. Pad the 2,051-entry KPool table to 2,176
# without hiding its three live tail entries behind base-table -1 padding.
captured = {}


def fake_generic(**kwargs):
    captured.clear()
    captured.update(kwargs)
    query = kwargs["query"]
    return query.new_full((*query.shape[:-1], kwargs["kv_lora_rank"]), 5)


flashinfer = ModuleType("flashinfer")
flashinfer.__path__ = []
mla = ModuleType("flashinfer.mla")
mla.trtllm_batch_decode_with_kv_cache_mla = fake_generic
flashinfer.mla = mla

indices = torch.arange(2051, dtype=torch.int32).view(1, -1).repeat(2, 1)
indices[:, 1536:2048] = -1
with patch.dict(sys.modules, {"flashinfer": flashinfer, "flashinfer.mla": mla}):
    output = flashinfer_sparse_mla_forward(
        q=torch.zeros((2, 32, 512), dtype=torch.bfloat16),
        kv_cache=torch.zeros((2, 64, 1, 528), dtype=torch.uint8),
        indices=indices,
        seq_lens=torch.tensor([1536, 1536], dtype=torch.int32),
        workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
        page_size=64,
        kv_cache_dim=528,
        qk_nope_head_dim=256,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        sm_scale=0.125,
        skip_softmax_threshold_scale_factor=None,
    )

block_tables = captured["block_tables"].squeeze(1)
assert tuple(captured["query"].shape) == (2, 1, 32, 512)
assert tuple(block_tables.shape) == (2, 2176)
assert block_tables[:, 2048:2051].tolist() == [
    [2048, 2049, 2050],
    [2048, 2049, 2050],
]
assert torch.all(block_tables[:, 2051:] == -1)
assert captured["seq_lens"].tolist() == [2176, 2176]
assert captured["sparse_mla_top_k_lens"].tolist() == [2176, 2176]
assert captured["max_seq_len"] == 2176
assert captured["sparse_mla_top_k"] == 2176
assert captured["kv_scale_format"] == "arbitrary_fp32"
assert tuple(captured["kv_cache"].shape) == (2, 1, 64, 528)
assert tuple(output.shape) == (2, 32, 512)
assert torch.all(output == 5)

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

assert captured["sparse_mla_top_k"] == 4
assert captured["seq_lens"].tolist() == [2, 3]
assert captured["sparse_mla_top_k_lens"] is None
assert tuple(generic_output.shape) == (2, 8, 512)

# The backend guard opens KPool tails only for the exact 528-byte no-RoPE
# geometry and remains fail-closed if any geometry field differs.
backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
backend.dsa_index_kpool = 4
backend.kv_cache_dim = 528
backend.qk_nope_head_dim = 256
backend.kv_lora_rank = 512
backend.qk_rope_head_dim = 0
backend._check_kpool_tail_backend(indices, SPARSE, "decode")

for attr, bad_value in (
    ("kv_cache_dim", 656),
    ("qk_nope_head_dim", 192),
    ("kv_lora_rank", 448),
    ("qk_rope_head_dim", 64),
):
    old_value = getattr(backend, attr)
    setattr(backend, attr, bad_value)
    try:
        backend._check_kpool_tail_backend(indices, SPARSE, "decode")
    except NotImplementedError:
        pass
    else:
        raise AssertionError(f"KPool guard unexpectedly accepted {attr}={bad_value}")
    setattr(backend, attr, old_value)

print("GLM-5.3 SM120 sparse-MLA architecture and no-RoPE KPool contracts OK")
