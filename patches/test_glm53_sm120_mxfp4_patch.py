import inspect
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.mxfp4 import (
    Mxfp4MoEMethod,
    _resolve_sm120_mxfp4_swiglu,
    _split_sm120_mxfp4_gate_up,
)


def config(**overrides):
    values = {
        "gemm1_alpha": None,
        "gemm1_beta": None,
        "gemm1_clamp_limit": None,
        "swiglu_limit": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# Per-expert GLM loading produces contiguous [gate; up] halves.
halves = torch.tensor([[[10], [11], [20], [21]]])
gate, up = _split_sm120_mxfp4_gate_up(halves, pairwise=False)
torch.testing.assert_close(gate, torch.tensor([[[10], [11]]]))
torch.testing.assert_close(up, torch.tensor([[[20], [21]]]))

# GPT-OSS's fused checkpoint produces pairwise [gate_i, up_i] rows.
pairwise = torch.tensor([[[10], [20], [11], [21]]])
gate, up = _split_sm120_mxfp4_gate_up(pairwise, pairwise=True)
torch.testing.assert_close(gate, torch.tensor([[[10], [11]]]))
torch.testing.assert_close(up, torch.tensor([[[20], [21]]]))

# GLM-5.3-Flash's clamped standard SwiGLU contract from config.json.
assert _resolve_sm120_mxfp4_swiglu(config(swiglu_limit=10.0)) == (1.0, 0.0, 10.0)

# Preserve GPT-OSS defaults and explicit overrides.
assert _resolve_sm120_mxfp4_swiglu(
    config(gemm1_alpha=1.702, gemm1_clamp_limit=7.0)
) == (1.702, 1.0, 7.0)
assert _resolve_sm120_mxfp4_swiglu(
    config(gemm1_alpha=1.5, gemm1_beta=0.25, gemm1_clamp_limit=6.0)
) == (1.5, 0.25, 6.0)

# The runner configuration belongs to the quantization method.  Reading it
# from the layer breaks the valid direct post-loader contract exercised by the
# vendor's GPT-OSS SM120 GPU regression and is not guaranteed by FusedMoE.
post_loader_source = inspect.getsource(Mxfp4MoEMethod._process_weights_for_sm120_cutlass)
assert "layer.moe_runner_config" not in post_loader_source
assert post_loader_source.count("self.moe_runner_config") == 2

print("GLM-5.3 SM120 MXFP4 layout and activation contract valid")
