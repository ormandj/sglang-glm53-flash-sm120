#!/usr/bin/env python3
"""Assert the SGLang resolver contract for routed-only MXFP4 + BF16 protection."""

import inspect

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration


TARGET = (
    r"re:^model\.language_model\.layers\."
    r"(?:[3-9]|[1-3][0-9]|4[0-5])\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)
IGNORE = [r"re:.*"]
CONFIG = {
    "config_groups": {
        "routed_experts": {
            "targets": [TARGET],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": False,
                "scale_dtype": "torch.uint8",
            },
            "input_activations": None,
        }
    },
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "ignore": IGNORE,
}


resolved = CompressedTensorsConfig.from_config(CONFIG)
assert resolved.ignore == IGNORE
assert resolved.quant_format == "mxfp4-pack-quantized"

# Ordinary target, shared-expert and vision linears must take the protected
# unquantized path instead of raising unmatched-target errors.
for prefix in (
    "model.layers.0.self_attn.o_proj",
    "model.layers.3.mlp.shared_experts.gate_proj",
    "visual.blocks.0.mlp.gate_proj",
    "lm_head",
):
    assert should_ignore_layer(prefix, resolved.ignore), prefix

# SGLang deliberately selects MXFP4 FusedMoE from the global format before
# scheme/ignore resolution. With shared-expert fusion disabled, every such
# module corresponds exactly to the routed experts serialized by the recipe.
assert resolved._is_mxfp4_moe("model.layers.3.mlp")
resolver_source = inspect.getsource(CompressedTensorsConfig.get_quant_method)
assert resolver_source.index("_is_mxfp4_moe") < resolver_source.index(
    "get_moe_scheme"
)

# GLM must honor the runtime flag before it appends the BF16 shared expert to
# the fused routed-expert buffer.
fusion_source = inspect.getsource(
    Glm5NextForConditionalGeneration.determine_num_fused_shared_experts
)
assert fusion_source.index("disable_shared_experts_fusion") < fusion_source.index(
    "shared_experts_fusion_disable_reason"
)

print("GLM-5.3 routed-only MXFP4 + protected BF16 runtime contract valid")
