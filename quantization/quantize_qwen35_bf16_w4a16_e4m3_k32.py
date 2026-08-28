#!/usr/bin/env python3
"""Produce the Qwen3.5-35B-A3B E4M3-K32 W4A16 kernel canary.

This is deliberately not the GLM production quantizer.  It converts the 40
fused routed-expert banks in a previously qualified BF16 Qwen VLM while leaving
vision, attention, dense/shared MLPs, and MTP in BF16.  Serving this artifact
proves the exact ModelOpt checkpoint schema, SGLang loader, FlashInfer packer,
and SM120 W4A16 kernel before a 744B GLM quantization is attempted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.kernels.quantization.gemm import nvfp4_fp8_scale_sweep
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor


GROUP_SIZE = 32
E2M1_MAX = 6.0
E4M3_MAX = 448.0
EXPECTED_MODELOPT_VERSION = "0.47.0rc0"
GATE_UP_SUFFIX = ".mlp.experts.gate_up_proj"
DOWN_SUFFIX = ".mlp.experts.down_proj"

IGNORE_MODULES = [
    "lm_head",
    "model.embed_tokens",
    "model.language_model.embed_tokens",
    "*.self_attn.*",
    "*.linear_attn.*",
    "*.mlp.gate",
    "*.mlp.gate_up_proj",
    "*.mlp.gate_proj",
    "*.mlp.up_proj",
    "*.mlp.down_proj",
    "*.mlp.shared_expert.*",
    "*.mlp.shared_experts.*",
    "model.visual.*",
    "visual.*",
    "*.visual.*",
    "mtp.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-fused-layers", type=int, default=40)
    parser.add_argument("--expected-experts", type=int, default=256)
    parser.add_argument("--expected-hidden-size", type=int, default=2048)
    parser.add_argument("--expected-intermediate-size", type=int, default=512)
    parser.add_argument("--expected-mtp-tensors", type=int, default=785)
    parser.add_argument("--max-shard-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--max-aggregate-relative-l2", type=float, default=0.15)
    parser.add_argument("--min-matrix-cosine", type=float, default=0.95)
    return parser.parse_args()


def nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash exact tensor bytes without dtype conversions."""
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def quantize_matrix(
    weight_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if weight_cpu.ndim != 2:
        raise ValueError(f"expected a matrix, got {tuple(weight_cpu.shape)}")
    if weight_cpu.shape[-1] % GROUP_SIZE:
        raise ValueError(f"K={weight_cpu.shape[-1]} is not divisible by {GROUP_SIZE}")
    if weight_cpu.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"unsupported source dtype: {weight_cpu.dtype}")

    weight = weight_cpu.to(device=device, dtype=torch.bfloat16, non_blocking=False).contiguous()
    global_amax = weight.abs().amax().to(torch.float32)
    if not torch.isfinite(global_amax) or global_amax.item() <= 0:
        raise ValueError(f"invalid global amax: {global_amax.item()}")

    blocked = weight.view(-1, GROUP_SIZE)
    best_amax = nvfp4_fp8_scale_sweep(blocked, global_amax, block_size=GROUP_SIZE)
    best_amax = best_amax.view(weight.shape[0], weight.shape[1] // GROUP_SIZE)
    per_block_scale = best_amax / E2M1_MAX
    # Match ModelOpt's static-export zero-block convention exactly. The packed
    # weights remain zero; the nonzero scale prevents underflow/NaN sentinels.
    per_block_scale[per_block_scale == 0] = 1.0
    block_scale = NVFP4QTensor._cast_per_block_scale_to_fp8(
        per_block_scale,
        global_amax / E2M1_MAX,
        fp8_max_for_normalization=E4M3_MAX,
    )
    global_scale = global_amax / (E2M1_MAX * E4M3_MAX)

    qtensor, emitted_scale, emitted_global = NVFP4QTensor.quantize(
        weight,
        GROUP_SIZE,
        weights_scaling_factor=block_scale,
        weights_scaling_factor_2=global_scale,
        try_tensorrt=False,
    )
    if emitted_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"unexpected scale dtype: {emitted_scale.dtype}")
    if emitted_scale.shape != block_scale.shape:
        raise ValueError(f"unexpected scale shape: {tuple(emitted_scale.shape)}")
    if qtensor._quantized_data.dtype != torch.uint8:
        raise TypeError(f"unexpected packed dtype: {qtensor._quantized_data.dtype}")
    if not torch.isfinite(emitted_scale.float()).all():
        raise ValueError("non-finite E4M3 block scale")
    if not torch.isfinite(emitted_global).all() or emitted_global.item() <= 0:
        raise ValueError(f"invalid emitted global scale: {emitted_global.item()}")

    dequant = qtensor.dequantize(
        dtype=torch.float32,
        scale=emitted_scale,
        double_scale=emitted_global,
        block_sizes={-1: GROUP_SIZE},
    )
    reference = weight.float()
    error = dequant - reference
    error_sq = error.square().sum(dtype=torch.float64).item()
    reference_sq = reference.square().sum(dtype=torch.float64).item()
    cosine = torch.nn.functional.cosine_similarity(
        dequant.reshape(1, -1), reference.reshape(1, -1), dim=1
    ).item()
    metrics = {
        "error_sq": error_sq,
        "reference_sq": reference_sq,
        "relative_l2": math.sqrt(error_sq / reference_sq),
        "cosine": cosine,
        "max_abs_error": error.abs().amax().item(),
    }

    packed_cpu = qtensor._quantized_data.detach().cpu().contiguous()
    scale_cpu = emitted_scale.detach().cpu().contiguous()
    global_cpu = emitted_global.detach().to(torch.float32).cpu().reshape(()).contiguous()
    del dequant, error, reference, qtensor, emitted_scale, emitted_global
    del best_amax, per_block_scale, block_scale, global_amax, global_scale, blocked, weight
    return packed_cpu, scale_cpu, global_cpu, metrics


class ShardWriter:
    def __init__(self, root: Path, max_bytes: int):
        self.root = root
        self.max_bytes = max_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.parts: list[Path] = []
        self.weight_map: dict[str, str] = {}
        self.total_size = 0
        self.category_bytes: dict[str, int] = defaultdict(int)
        self.tensor_specs: dict[
            str, tuple[tuple[int, ...], torch.dtype, str, str | None]
        ] = {}

    def add(self, name: str, tensor: torch.Tensor, category: str) -> None:
        if name in self.weight_map or name in self.pending:
            raise ValueError(f"duplicate output tensor: {name}")
        tensor = tensor.detach().cpu().contiguous()
        size = nbytes(tensor)
        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += size
        self.total_size += size
        self.category_bytes[category] += size
        digest = sha256_tensor(tensor) if category == "protected_source_precision" else None
        self.tensor_specs[name] = (tuple(tensor.shape), tensor.dtype, category, digest)

    def flush(self) -> None:
        if not self.pending:
            return
        part = self.root / f"part-{len(self.parts) + 1:05d}.safetensors"
        save_file(self.pending, str(part), metadata={"format": "pt"})
        for name in self.pending:
            self.weight_map[name] = part.name
        self.parts.append(part)
        self.pending.clear()
        self.pending_bytes = 0
        gc.collect()

    def finish(self) -> dict[str, Any]:
        self.flush()
        count = len(self.parts)
        rename: dict[str, str] = {}
        for number, old_path in enumerate(self.parts, 1):
            new_name = f"model-{number:05d}-of-{count:05d}.safetensors"
            os.replace(old_path, self.root / new_name)
            rename[old_path.name] = new_name
        final_map = {name: rename[file_name] for name, file_name in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total_size}, "weight_map": final_map}
        json_dump(self.root / "model.safetensors.index.json", index)
        return index


def validate_output_roundtrip(
    root: Path,
    writer: ShardWriter,
    output_index: dict[str, Any],
) -> dict[str, int]:
    expected_names = set(writer.tensor_specs)
    indexed_names = set(output_index["weight_map"])
    if indexed_names != expected_names:
        missing = sorted(expected_names - indexed_names)[:10]
        extra = sorted(indexed_names - expected_names)[:10]
        raise ValueError(f"output index mismatch: missing={missing} extra={extra}")

    seen: set[str] = set()
    category_counts: dict[str, int] = defaultdict(int)
    for shard_path in sorted(root.glob("model-*-of-*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ValueError(f"duplicate tensor in output shards: {name}")
                seen.add(name)
                if output_index["weight_map"].get(name) != shard_path.name:
                    raise ValueError(f"index points {name} at the wrong shard")
                tensor = handle.get_tensor(name)
                shape, dtype, category, expected_digest = writer.tensor_specs[name]
                if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(
                        f"round-trip mismatch for {name}: "
                        f"{tuple(tensor.shape)}/{tensor.dtype} != {shape}/{dtype}"
                    )
                if expected_digest is not None and sha256_tensor(tensor) != expected_digest:
                    raise ValueError(f"protected tensor changed during round trip: {name}")
                if category in (
                    "routed_expert_e4m3_scale",
                    "routed_expert_global_scale",
                    "routed_expert_input_scale",
                ) and not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"non-finite quantization scale in {name}")
                if category == "routed_expert_global_scale" and not torch.all(tensor > 0):
                    raise ValueError(f"non-positive global scale in {name}")
                if category == "routed_expert_input_scale" and not torch.equal(
                    tensor, torch.ones_like(tensor)
                ):
                    raise ValueError(f"non-neutral W4A16 input scale in {name}")
                category_counts[category] += 1
    if seen != expected_names:
        missing = sorted(expected_names - seen)[:10]
        extra = sorted(seen - expected_names)[:10]
        raise ValueError(f"output shard traversal mismatch: missing={missing} extra={extra}")
    return dict(sorted(category_counts.items()))


def add_quantized_gate_up(
    writer: ShardWriter,
    source_name: str,
    tensor: torch.Tensor,
    expected_experts: int,
    expected_hidden_size: int,
    expected_intermediate_size: int,
    device: torch.device,
    metric_rows: list[dict[str, Any]],
) -> None:
    expected_shape = (
        expected_experts,
        2 * expected_intermediate_size,
        expected_hidden_size,
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"unexpected fused gate/up shape for {source_name}: {tuple(tensor.shape)}")
    prefix = source_name[: -len(GATE_UP_SUFFIX)]
    half = tensor.shape[1] // 2
    one = torch.tensor(1.0, dtype=torch.float32)
    for expert in range(expected_experts):
        packed, scale, global_scale, metrics = quantize_matrix(tensor[expert], device)
        gate_packed, up_packed = packed.split(half, dim=0)
        gate_scale, up_scale = scale.split(half, dim=0)
        expert_prefix = f"{prefix}.mlp.experts.{expert}"
        for projection, proj_packed, proj_scale in (
            ("gate_proj", gate_packed, gate_scale),
            ("up_proj", up_packed, up_scale),
        ):
            base = f"{expert_prefix}.{projection}"
            writer.add(f"{base}.weight", proj_packed, "routed_expert_packed")
            writer.add(f"{base}.weight_scale", proj_scale, "routed_expert_e4m3_scale")
            writer.add(f"{base}.weight_scale_2", global_scale.clone(), "routed_expert_global_scale")
            writer.add(f"{base}.input_scale", one.clone(), "routed_expert_input_scale")
        metric_rows.append({"source": source_name, "expert": expert, **metrics})
    del tensor
    torch.cuda.empty_cache()


def add_quantized_down(
    writer: ShardWriter,
    source_name: str,
    tensor: torch.Tensor,
    expected_experts: int,
    expected_hidden_size: int,
    expected_intermediate_size: int,
    device: torch.device,
    metric_rows: list[dict[str, Any]],
) -> None:
    expected_shape = (
        expected_experts,
        expected_hidden_size,
        expected_intermediate_size,
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"unexpected fused down shape for {source_name}: {tuple(tensor.shape)}")
    prefix = source_name[: -len(DOWN_SUFFIX)]
    one = torch.tensor(1.0, dtype=torch.float32)
    for expert in range(expected_experts):
        packed, scale, global_scale, metrics = quantize_matrix(tensor[expert], device)
        base = f"{prefix}.mlp.experts.{expert}.down_proj"
        writer.add(f"{base}.weight", packed, "routed_expert_packed")
        writer.add(f"{base}.weight_scale", scale, "routed_expert_e4m3_scale")
        writer.add(f"{base}.weight_scale_2", global_scale, "routed_expert_global_scale")
        writer.add(f"{base}.input_scale", one.clone(), "routed_expert_input_scale")
        metric_rows.append({"source": source_name, "expert": expert, **metrics})
    del tensor
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    started = time.time()
    source = args.source.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {incomplete}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.parent != incomplete.parent:
        raise AssertionError("atomic publish requires a same-filesystem sibling")

    installed_modelopt = importlib.metadata.version("nvidia-modelopt")
    if installed_modelopt != EXPECTED_MODELOPT_VERSION:
        raise RuntimeError(
            f"ModelOpt {installed_modelopt} is not pinned {EXPECTED_MODELOPT_VERSION}"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the MSE scale sweep requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError(f"expected SM120/SM121, got {capability}")

    source_revision = (source / ".source-revision").read_text().strip()
    if source_revision != args.expected_source_revision:
        raise ValueError(f"source revision mismatch: {source_revision}")
    source_index_path = source / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text())
    source_weight_map: dict[str, str] = source_index["weight_map"]
    gate_up_names = sorted(name for name in source_weight_map if name.endswith(GATE_UP_SUFFIX))
    down_names = sorted(name for name in source_weight_map if name.endswith(DOWN_SUFFIX))
    if len(gate_up_names) != args.expected_fused_layers or len(down_names) != args.expected_fused_layers:
        raise ValueError(
            f"expected {args.expected_fused_layers} fused layers, got "
            f"gate_up={len(gate_up_names)} down={len(down_names)}"
        )

    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]:
        raise ValueError(f"unexpected architecture: {config.get('architectures')}")
    if config.get("quantization_config"):
        raise ValueError("canary source is already quantized")
    if (config.get("vision_config") or {}).get("depth") != 27:
        raise ValueError("missing or unexpected Qwen3.5 vision tower")
    text_config = config["text_config"]
    configured_experts = int(text_config["num_experts"])
    if configured_experts != args.expected_experts:
        raise ValueError(f"expected {args.expected_experts} experts, got {configured_experts}")
    source_contract = {
        "num_hidden_layers": args.expected_fused_layers,
        "hidden_size": args.expected_hidden_size,
        "moe_intermediate_size": args.expected_intermediate_size,
        "num_experts_per_tok": 8,
        "mtp_num_hidden_layers": 1,
    }
    for field, expected in source_contract.items():
        if text_config.get(field) != expected:
            raise ValueError(
                f"unexpected text_config.{field}: {text_config.get(field)!r} != {expected!r}"
            )
    mtp_names = sorted(name for name in source_weight_map if "mtp" in name)
    if len(mtp_names) != args.expected_mtp_tensors or not all(
        name.startswith("mtp.") for name in mtp_names
    ):
        raise ValueError(
            f"expected {args.expected_mtp_tensors} mtp.* tensors, got {len(mtp_names)}"
        )
    quant_config = {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": GROUP_SIZE,
                    "dynamic": False,
                    "symmetric": True,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "quant_algo": "W4A16_NVFP4",
        "group_size": GROUP_SIZE,
        "kv_cache_scheme": None,
        "quant_method": "modelopt",
        "ignore": IGNORE_MODULES,
        "producer": {
            "name": "modelopt",
            "version": installed_modelopt,
            "commit": os.environ.get("MODELOPT_BUILD_COMMIT", "unknown"),
        },
    }

    incomplete.mkdir(parents=False)
    for child in source.iterdir():
        if child.name.startswith("model") and child.suffix in (".safetensors", ".json"):
            continue
        if child.name in ("config.json", ".download-complete"):
            continue
        destination = incomplete / child.name
        if child.is_file():
            shutil.copy2(child, destination)
        elif child.is_dir():
            shutil.copytree(child, destination, symlinks=True)

    writer = ShardWriter(incomplete, args.max_shard_bytes)
    metrics: list[dict[str, Any]] = []
    processed_source_names: set[str] = set()
    source_shards = sorted(set(source_weight_map.values()))
    for shard_number, shard_name in enumerate(source_shards, 1):
        shard_path = source / shard_name
        print(f"[{shard_number}/{len(source_shards)}] {shard_name}", flush=True)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if name.endswith(GATE_UP_SUFFIX):
                    add_quantized_gate_up(
                        writer,
                        name,
                        tensor,
                        args.expected_experts,
                        args.expected_hidden_size,
                        args.expected_intermediate_size,
                        device,
                        metrics,
                    )
                elif name.endswith(DOWN_SUFFIX):
                    add_quantized_down(
                        writer,
                        name,
                        tensor,
                        args.expected_experts,
                        args.expected_hidden_size,
                        args.expected_intermediate_size,
                        device,
                        metrics,
                    )
                else:
                    writer.add(name, tensor, "protected_source_precision")
                processed_source_names.add(name)
        gc.collect()
    if processed_source_names != set(source_weight_map):
        missing = sorted(set(source_weight_map) - processed_source_names)[:10]
        extra = sorted(processed_source_names - set(source_weight_map))[:10]
        raise ValueError(f"source traversal mismatch: missing={missing} extra={extra}")

    output_index = writer.finish()
    expected_quantized_matrices = (
        args.expected_fused_layers * args.expected_experts * 2
    )
    if len(metrics) != expected_quantized_matrices:
        raise ValueError(
            f"expected {expected_quantized_matrices} quantized matrices, got {len(metrics)}"
        )
    error_sq = sum(row["error_sq"] for row in metrics)
    reference_sq = sum(row["reference_sq"] for row in metrics)
    aggregate_relative_l2 = math.sqrt(error_sq / reference_sq)
    min_cosine = min(row["cosine"] for row in metrics)
    if not math.isfinite(aggregate_relative_l2) or (
        aggregate_relative_l2 > args.max_aggregate_relative_l2
    ):
        raise ValueError(
            f"aggregate relative L2 {aggregate_relative_l2} exceeds "
            f"{args.max_aggregate_relative_l2}"
        )
    if not math.isfinite(min_cosine) or min_cosine < args.min_matrix_cosine:
        raise ValueError(f"minimum matrix cosine {min_cosine} is below {args.min_matrix_cosine}")
    roundtrip_counts = validate_output_roundtrip(incomplete, writer, output_index)
    config["quantization_config"] = quant_config
    json_dump(incomplete / "config.json", config)

    summary = {
        "schema": 1,
        "purpose": "Qwen BF16 platform canary for the GLM E4M3-K32 W4A16 path",
        "source": str(source),
        "source_revision": source_revision,
        "source_index_sha256": sha256_file(source_index_path),
        "output": str(output),
        "modelopt_version": installed_modelopt,
        "modelopt_commit": os.environ.get("MODELOPT_BUILD_COMMIT", "unknown"),
        "sglang_commit": os.environ.get("SGLANG_BUILD_COMMIT", "unknown"),
        "flashinfer_commit": os.environ.get("FLASHINFER_BUILD_COMMIT", "unknown"),
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
        },
        "recipe": quant_config,
        "selection": {
            "fused_gate_up_layers": len(gate_up_names),
            "fused_down_layers": len(down_names),
            "experts_per_layer": args.expected_experts,
            "quantized_matrices": len(metrics),
            "protected_mtp_tensors": len(mtp_names),
        },
        "reconstruction": {
            "aggregate_relative_l2": aggregate_relative_l2,
            "mean_relative_l2": sum(row["relative_l2"] for row in metrics) / len(metrics),
            "max_relative_l2": max(row["relative_l2"] for row in metrics),
            "min_cosine": min_cosine,
            "gates": {
                "max_aggregate_relative_l2": args.max_aggregate_relative_l2,
                "min_matrix_cosine": args.min_matrix_cosine,
            },
        },
        "roundtrip_tensor_counts": roundtrip_counts,
        "tensor_bytes": dict(sorted(writer.category_bytes.items())),
        "total_tensor_bytes": output_index["metadata"]["total_size"],
        "tensor_count": len(output_index["weight_map"]),
        "elapsed_seconds": time.time() - started,
    }
    json_dump(incomplete / "quantization-manifest.json", summary)
    json_dump(incomplete / "reconstruction-metrics.json", metrics)

    shard_hashes = {
        path.name: sha256_file(path)
        for path in sorted(incomplete.glob("model-*-of-*.safetensors"))
    }
    json_dump(incomplete / "model-sha256.json", shard_hashes)
    (incomplete / ".quantization-complete").write_text("complete\n")
    os.replace(incomplete, output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
