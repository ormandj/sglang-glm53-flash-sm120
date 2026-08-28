#!/usr/bin/env python3
"""Compare real GLM BF16 experts with the published MXFP4 SM120 runtime ABI.

This is deliberately an actual-weight test.  It distinguishes quantization
error from packed-layout/runtime error by comparing three outputs:

1. the BF16 teacher weights;
2. the published weights decompressed by compressed-tensors; and
3. the published packed bytes executed by FlashInfer's SM120 MXFP8-by-MXFP4
   fused-MoE kernel, using the same scale interleave and SwiGLU contract as
   SGLang.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import MXFP4A16
from flashinfer import block_scale_interleave, mxfp8_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from safetensors import safe_open


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class IndexedCheckpoint:
    def __init__(self, root: Path):
        self.root = root
        index_path = root / "model.safetensors.index.json"
        self.index = json.loads(index_path.read_text())["weight_map"]

    def load(self, name: str) -> torch.Tensor:
        shard = self.root / self.index[name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


def module_name(layer: int, expert: int, projection: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}"
    )


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float())
    return float((numerator / denominator).item())


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float().reshape(-1)
    expected_f = expected.float().reshape(-1)
    return {
        "relative_l2": relative_l2(actual_f, expected_f),
        "cosine": float(F.cosine_similarity(actual_f, expected_f, dim=0).item()),
        "max_abs": float((actual_f - expected_f).abs().max().item()),
        "expected_rms": float(expected_f.square().mean().sqrt().item()),
    }


def reference_moe(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    routing_weights: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    x_f = x.float()
    output = torch.zeros_like(x_f)
    for expert in range(gate.shape[0]):
        gate_out = x_f @ gate[expert].float().T
        up_out = x_f @ up[expert].float().T
        gate_out = torch.clamp_max(gate_out, limit)
        up_out = torch.clamp(up_out, -limit, limit)
        hidden = F.silu(gate_out) * up_out
        expert_out = hidden @ down[expert].float().T
        output.add_(expert_out * routing_weights[:, expert : expert + 1])
    return output


def run_kernel(
    x: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_up: torch.Tensor,
    packed_down: torch.Tensor,
    gate_scale: torch.Tensor,
    up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    gate_first_control: bool,
    limit: float,
) -> torch.Tensor:
    if gate_first_control:
        w13 = torch.cat((packed_gate, packed_up), dim=1).contiguous()
        w13_scale = torch.cat((gate_scale, up_scale), dim=1).contiguous()
    else:
        # FlashInfer's FC1 ABI consumes [up; gate].
        w13 = torch.cat((packed_up, packed_gate), dim=1).contiguous()
        w13_scale = torch.cat((up_scale, gate_scale), dim=1).contiguous()

    packed_down = packed_down.contiguous()
    down_scale = down_scale.contiguous()
    w13_scale = block_scale_interleave(w13_scale).reshape_as(w13_scale)
    w2_scale = block_scale_interleave(down_scale).reshape_as(down_scale)
    x_quant, x_scale = mxfp8_quantize(
        x, is_sf_swizzled_layout=True, alignment=32
    )

    experts = packed_gate.shape[0]
    selected = torch.arange(experts, device=x.device, dtype=torch.int32)
    selected = selected.unsqueeze(0).expand(x.shape[0], -1).contiguous()
    global_scale = torch.ones(experts, dtype=torch.float32, device=x.device)
    zeros_w13 = torch.zeros(
        experts, w13.shape[1], dtype=torch.bfloat16, device=x.device
    )
    zeros_w2 = torch.zeros(
        experts, packed_down.shape[1], dtype=torch.bfloat16, device=x.device
    )
    output = torch.empty_like(x)
    cutlass_fused_moe(
        input=x_quant,
        token_selected_experts=selected,
        token_final_scales=routing_weights,
        fc1_expert_weights=w13.view(torch.int64),
        fc2_expert_weights=packed_down.view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[
            w13_scale.view(torch.int32),
            global_scale,
            w2_scale.view(torch.int32),
            global_scale,
        ],
        input_sf=x_scale,
        fc1_expert_biases=zeros_w13,
        fc2_expert_biases=zeros_w2,
        swiglu_alpha=torch.ones(experts, dtype=torch.float32, device=x.device),
        swiglu_beta=torch.zeros(experts, dtype=torch.float32, device=x.device),
        swiglu_limit=torch.full(
            (experts,), limit, dtype=torch.float32, device=x.device
        ),
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=x.shape[0],
        output=output,
    )
    torch.cuda.synchronize()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--routed-scaling-factor", type=float, default=2.5)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit("this regression requires one SM120 CUDA GPU")

    bf16 = IndexedCheckpoint(args.bf16)
    quant = IndexedCheckpoint(args.quant)
    weights = MXFP4A16["weights"].model_copy()
    scheme = QuantizationScheme(targets=["Linear"], weights=weights)

    source: dict[str, list[torch.Tensor]] = {name: [] for name in PROJECTIONS}
    restored: dict[str, list[torch.Tensor]] = {name: [] for name in PROJECTIONS}
    packed: dict[str, list[torch.Tensor]] = {name: [] for name in PROJECTIONS}
    scales: dict[str, list[torch.Tensor]] = {name: [] for name in PROJECTIONS}
    weight_errors: dict[str, list[float]] = {name: [] for name in PROJECTIONS}

    for expert in range(args.experts):
        for projection in PROJECTIONS:
            module = module_name(args.layer, expert, projection)
            original = bf16.load(module + ".weight")
            packed_weight = quant.load(module + ".weight_packed")
            scale = quant.load(module + ".weight_scale")
            decompressed = MXFP4PackedCompressor.decompress(
                {"weight_packed": packed_weight, "weight_scale": scale}, scheme
            )["weight"]
            if decompressed.shape != original.shape:
                raise RuntimeError(
                    f"shape mismatch for {module}: "
                    f"{tuple(decompressed.shape)} != {tuple(original.shape)}"
                )
            error = relative_l2(decompressed, original)
            if not math.isfinite(error):
                raise RuntimeError(f"non-finite weight error for {module}")
            source[projection].append(original)
            restored[projection].append(decompressed)
            packed[projection].append(packed_weight)
            scales[projection].append(scale)
            weight_errors[projection].append(error)

    for collection in (source, restored, packed, scales):
        for projection in PROJECTIONS:
            collection[projection] = torch.stack(collection[projection]).cuda()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    hidden_size = source["gate_proj"].shape[-1]
    x = torch.randn(
        args.tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    routing = torch.rand(
        args.tokens,
        args.experts,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    routing = routing / routing.sum(dim=-1, keepdim=True)
    routing = routing * args.routed_scaling_factor

    with torch.no_grad():
        teacher = reference_moe(
            x,
            source["gate_proj"],
            source["up_proj"],
            source["down_proj"],
            routing,
            args.swiglu_limit,
        )
        quant_reference = reference_moe(
            x,
            restored["gate_proj"],
            restored["up_proj"],
            restored["down_proj"],
            routing,
            args.swiglu_limit,
        )
        kernel = run_kernel(
            x,
            packed["gate_proj"],
            packed["up_proj"],
            packed["down_proj"],
            scales["gate_proj"],
            scales["up_proj"],
            scales["down_proj"],
            routing,
            gate_first_control=False,
            limit=args.swiglu_limit,
        )
        intermediate = packed["gate_proj"].shape[1]
        if intermediate % args.tp_size != 0:
            raise RuntimeError(
                f"intermediate size {intermediate} is not divisible by TP "
                f"size {args.tp_size}"
            )
        shard = intermediate // args.tp_size
        tp_kernel = torch.zeros_like(kernel)
        for rank in range(args.tp_size):
            start = rank * shard
            stop = start + shard
            tp_kernel.add_(
                run_kernel(
                    x,
                    packed["gate_proj"][:, start:stop, :],
                    packed["up_proj"][:, start:stop, :],
                    packed["down_proj"][:, :, start // 2 : stop // 2],
                    scales["gate_proj"][:, start:stop, :],
                    scales["up_proj"][:, start:stop, :],
                    scales["down_proj"][:, :, start // 32 : stop // 32],
                    routing,
                    gate_first_control=False,
                    limit=args.swiglu_limit,
                )
            )
        swapped_control = run_kernel(
            x,
            packed["gate_proj"],
            packed["up_proj"],
            packed["down_proj"],
            scales["gate_proj"],
            scales["up_proj"],
            scales["down_proj"],
            routing,
            gate_first_control=True,
            limit=args.swiglu_limit,
        )

    result = {
        "schema": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "bf16": str(args.bf16),
        "quant": str(args.quant),
        "layer": args.layer,
        "experts": args.experts,
        "tokens": args.tokens,
        "tp_size": args.tp_size,
        "swiglu_limit": args.swiglu_limit,
        "routed_scaling_factor": args.routed_scaling_factor,
        "weight_relative_l2": {
            projection: {
                "minimum": min(errors),
                "mean": sum(errors) / len(errors),
                "maximum": max(errors),
            }
            for projection, errors in weight_errors.items()
        },
        "decompressed_quant_vs_bf16": metrics(quant_reference, teacher),
        "sm120_kernel_vs_decompressed_quant": metrics(kernel, quant_reference),
        "sm120_kernel_vs_bf16": metrics(kernel, teacher),
        "sm120_tp_split_reduce_vs_decompressed_quant": metrics(
            tp_kernel, quant_reference
        ),
        "sm120_tp_split_reduce_vs_full_kernel": metrics(tp_kernel, kernel),
        "gate_first_control_vs_decompressed_quant": metrics(
            swapped_control, quant_reference
        ),
    }
    if not all(
        math.isfinite(value)
        for comparison in (
            result["decompressed_quant_vs_bf16"],
            result["sm120_kernel_vs_decompressed_quant"],
            result["sm120_kernel_vs_bf16"],
            result["sm120_tp_split_reduce_vs_decompressed_quant"],
            result["sm120_tp_split_reduce_vs_full_kernel"],
            result["gate_first_control_vs_decompressed_quant"],
        )
        for value in comparison.values()
    ):
        raise RuntimeError("non-finite output metric")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
