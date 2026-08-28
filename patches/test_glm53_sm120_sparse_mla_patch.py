"""Build-time contract for the GLM-5.3 SM120 sparse-MLA allowlist fix."""

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _GLM_DSA_MODEL_ARCHS,
    _validate_flashinfer_sparse_mla_backend,
)


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

print("GLM-5.3 SM120 sparse-MLA architecture contract OK")
