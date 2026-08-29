#!/usr/bin/env python3
"""Produce the GLM-5.3-Flash E4M3-K32 W4A16 candidate.

Only the 288 routed experts in layers 3 through 45 are quantized.  Layer 45 is
the native NextN/MTP layer and intentionally uses the same representation as
the target experts.  Vision, embeddings/head, attention/KDA/mHC, routers,
dense MLPs, and shared experts remain byte-identical to the BF16 teacher.

The output is published atomically only after exact source-contract,
reconstruction, tensor-roundtrip, and whole-shard hash checks pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
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
EXPECTED_MODELOPT_COMMIT = "022767c7ab3d7d36211affd85e5c496770cde768"
EXPECTED_SGLANG_COMMIT = "42a56dc505f775d6f54e9d27a9b57c66023420a0"
EXPECTED_FLASHINFER_COMMIT = "008122fa75c7a27c839feea57a6ef8e8846fa265"
EXPECTED_SOURCE_REVISION = "f12e0fe1f6b2ea274c11a569582edfd99d993c5e"
EXPECTED_SOURCE_BYTES = 642_646_653_816
EXPECTED_SOURCE_TENSORS = 38_770
EXPECTED_SOURCE_SHARDS = 120
EXPECTED_LAYERS = tuple(range(3, 46))
EXPECTED_EXPERTS = 288
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_INTERMEDIATE_SIZE = 2048
EXPECTED_SHARED_TENSORS = 129
EXPECTED_VISION_TENSORS = 347

EXPERT_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)

# The serialized ModelOpt config applies FP4 to every Linear/FusedMoE unless a
# module is excluded.  These patterns protect every non-routed module while
# deliberately NOT excluding model.layers.45 as a whole: doing so would build
# the entire native draft layer unquantized, including its 288-expert bank.
IGNORE_MODULES = [
    "lm_head",
    "model.embed_tokens",
    "model.language_model.embed_tokens",
    "*.embed_tokens",
    "*.self_attn.*",
    "*.mlp.gate",
    "*.mlp.gate_up_proj",
    "*.mlp.gate_proj",
    "*.mlp.up_proj",
    "*.mlp.down_proj",
    "*.mlp.shared_expert.*",
    "*.mlp.shared_experts.*",
    "*.eh_proj",
    "*.shared_head.*",
    "model.visual.*",
    "visual.*",
    "*.visual.*",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-source-revision", default=EXPECTED_SOURCE_REVISION)
    parser.add_argument("--max-shard-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--max-aggregate-relative-l2", type=float, default=0.15)
    parser.add_argument("--min-matrix-cosine", type=float, default=0.95)
    parser.add_argument("--preflight-only", action="store_true")
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
    tensor = tensor.detach().cpu().contiguous()
    payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_toolchain() -> dict[str, str]:
    identities = {
        "modelopt_version": importlib.metadata.version("nvidia-modelopt"),
        "modelopt_commit": os.environ.get("MODELOPT_BUILD_COMMIT", ""),
        "sglang_commit": os.environ.get("SGLANG_BUILD_COMMIT", ""),
        "flashinfer_commit": os.environ.get("FLASHINFER_BUILD_COMMIT", ""),
    }
    expected = {
        "modelopt_version": EXPECTED_MODELOPT_VERSION,
        "modelopt_commit": EXPECTED_MODELOPT_COMMIT,
        "sglang_commit": EXPECTED_SGLANG_COMMIT,
        "flashinfer_commit": EXPECTED_FLASHINFER_COMMIT,
    }
    if identities != expected:
        raise RuntimeError(f"quantization toolchain mismatch: {identities} != {expected}")
    return identities


def quantize_matrix(
    weight_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if weight_cpu.ndim != 2:
        raise ValueError(f"expected matrix, got {tuple(weight_cpu.shape)}")
    if weight_cpu.shape[-1] % GROUP_SIZE:
        raise ValueError(f"K={weight_cpu.shape[-1]} is not divisible by {GROUP_SIZE}")
    if weight_cpu.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 teacher weight, got {weight_cpu.dtype}")

    weight = weight_cpu.to(device=device, dtype=torch.bfloat16).contiguous()
    global_amax = weight.abs().amax().to(torch.float32)
    if not torch.isfinite(global_amax) or global_amax.item() <= 0:
        raise ValueError(f"invalid global amax: {global_amax.item()}")

    blocked = weight.view(-1, GROUP_SIZE)
    best_amax = nvfp4_fp8_scale_sweep(blocked, global_amax, block_size=GROUP_SIZE)
    best_amax = best_amax.view(weight.shape[0], weight.shape[1] // GROUP_SIZE)
    per_block_scale = best_amax / E2M1_MAX
    # Match ModelOpt's static-export zero-block sentinel exactly.
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
    if qtensor._quantized_data.dtype != torch.uint8:
        raise TypeError(f"unexpected packed dtype: {qtensor._quantized_data.dtype}")
    if emitted_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"unexpected scale dtype: {emitted_scale.dtype}")
    if tuple(emitted_scale.shape) != (
        weight.shape[0],
        weight.shape[1] // GROUP_SIZE,
    ):
        raise ValueError(f"unexpected scale shape: {tuple(emitted_scale.shape)}")
    if not torch.isfinite(emitted_scale.float()).all():
        raise ValueError("non-finite E4M3 scale")
    if not torch.isfinite(emitted_global).all() or emitted_global.item() <= 0:
        raise ValueError(f"invalid global scale: {emitted_global.item()}")

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
    metrics = {
        "error_sq": error_sq,
        "reference_sq": reference_sq,
        "relative_l2": math.sqrt(error_sq / reference_sq),
        "cosine": torch.nn.functional.cosine_similarity(
            dequant.reshape(1, -1), reference.reshape(1, -1), dim=1
        ).item(),
        "max_abs_error": error.abs().amax().item(),
    }

    packed_cpu = qtensor._quantized_data.detach().cpu().contiguous()
    scale_cpu = emitted_scale.detach().cpu().contiguous()
    global_cpu = emitted_global.detach().to(torch.float32).cpu().reshape(()).contiguous()
    del dequant, error, reference, qtensor, emitted_scale, emitted_global
    del blocked, best_amax, per_block_scale, block_scale, global_amax, global_scale, weight
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
        digest = sha256_tensor(tensor) if category == "protected_bf16_or_fp32" else None
        self.tensor_specs[name] = (tuple(tensor.shape), tensor.dtype, category, digest)

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.root / f"part-{len(self.parts) + 1:05d}.safetensors"
        save_file(self.pending, str(path), metadata={"format": "pt"})
        for name in self.pending:
            self.weight_map[name] = path.name
        self.parts.append(path)
        self.pending.clear()
        self.pending_bytes = 0
        gc.collect()

    def finish(self) -> dict[str, Any]:
        self.flush()
        count = len(self.parts)
        renamed: dict[str, str] = {}
        for number, old_path in enumerate(self.parts, 1):
            new_name = f"model-{number:05d}-of-{count:05d}.safetensors"
            os.replace(old_path, self.root / new_name)
            renamed[old_path.name] = new_name
        final_map = {name: renamed[path] for name, path in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total_size}, "weight_map": final_map}
        json_dump(self.root / "model.safetensors.index.json", index)
        return index


def add_projection(
    writer: ShardWriter,
    base: str,
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    writer.add(f"{base}.weight", packed, "routed_expert_packed")
    writer.add(f"{base}.weight_scale", scale, "routed_expert_e4m3_scale")
    writer.add(
        f"{base}.weight_scale_2",
        global_scale.clone(),
        "routed_expert_global_scale",
    )
    writer.add(
        f"{base}.input_scale",
        torch.tensor(1.0, dtype=torch.float32),
        "routed_expert_input_scale",
    )


def quantize_gate_up(
    writer: ShardWriter,
    gate_name: str,
    gate: torch.Tensor,
    up_name: str,
    up: torch.Tensor,
    device: torch.device,
    metrics: list[dict[str, Any]],
) -> None:
    expected = (EXPECTED_INTERMEDIATE_SIZE, EXPECTED_HIDDEN_SIZE)
    if tuple(gate.shape) != expected or tuple(up.shape) != expected:
        raise ValueError(
            f"unexpected gate/up shapes: {gate_name}={tuple(gate.shape)}, "
            f"{up_name}={tuple(up.shape)}"
        )
    joint = torch.cat((gate, up), dim=0)
    packed, scale, global_scale, metric = quantize_matrix(joint, device)
    gate_packed, up_packed = packed.split(EXPECTED_INTERMEDIATE_SIZE, dim=0)
    gate_scale, up_scale = scale.split(EXPECTED_INTERMEDIATE_SIZE, dim=0)
    add_projection(writer, gate_name.removesuffix(".weight"), gate_packed, gate_scale, global_scale)
    add_projection(writer, up_name.removesuffix(".weight"), up_packed, up_scale, global_scale)
    match = EXPERT_PATTERN.fullmatch(gate_name)
    assert match is not None
    metrics.append(
        {
            "source": [gate_name, up_name],
            "layer": int(match.group(1)),
            "expert": int(match.group(2)),
            "calibration_unit": "joint_gate_up",
            **metric,
        }
    )
    del joint, packed, scale, global_scale, gate, up
    torch.cuda.empty_cache()


def quantize_down(
    writer: ShardWriter,
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
    metrics: list[dict[str, Any]],
) -> None:
    expected = (EXPECTED_HIDDEN_SIZE, EXPECTED_INTERMEDIATE_SIZE)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"unexpected down shape for {name}: {tuple(tensor.shape)}")
    packed, scale, global_scale, metric = quantize_matrix(tensor, device)
    add_projection(writer, name.removesuffix(".weight"), packed, scale, global_scale)
    match = EXPERT_PATTERN.fullmatch(name)
    assert match is not None
    metrics.append(
        {
            "source": [name],
            "layer": int(match.group(1)),
            "expert": int(match.group(2)),
            "calibration_unit": "down",
            **metric,
        }
    )
    del packed, scale, global_scale, tensor
    torch.cuda.empty_cache()


def validate_output_roundtrip(
    root: Path,
    writer: ShardWriter,
    output_index: dict[str, Any],
) -> dict[str, int]:
    expected_names = set(writer.tensor_specs)
    if set(output_index["weight_map"]) != expected_names:
        raise ValueError("output index tensor namespace mismatch")
    seen: set[str] = set()
    category_counts: dict[str, int] = defaultdict(int)
    for shard_path in sorted(root.glob("model-*-of-*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ValueError(f"duplicate tensor in output: {name}")
                seen.add(name)
                if output_index["weight_map"].get(name) != shard_path.name:
                    raise ValueError(f"output index points {name} at the wrong shard")
                tensor = handle.get_tensor(name)
                shape, dtype, category, expected_digest = writer.tensor_specs[name]
                if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(f"round-trip shape/dtype mismatch for {name}")
                if expected_digest is not None and sha256_tensor(tensor) != expected_digest:
                    raise ValueError(f"protected tensor changed: {name}")
                if category.endswith("scale") and not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"non-finite quantization scale: {name}")
                if category == "routed_expert_global_scale" and not torch.all(tensor > 0):
                    raise ValueError(f"non-positive global scale: {name}")
                if category == "routed_expert_input_scale" and not torch.equal(
                    tensor, torch.ones_like(tensor)
                ):
                    raise ValueError(f"non-neutral W4A16 input scale: {name}")
                category_counts[category] += 1
    if seen != expected_names:
        raise ValueError("output shard traversal tensor namespace mismatch")
    return dict(sorted(category_counts.items()))


def main() -> None:
    args = parse_args()
    started = time.time()
    toolchain = validate_toolchain()
    source = args.source.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {incomplete}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.parent != incomplete.parent:
        raise AssertionError("atomic publish requires a same-filesystem sibling")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ModelOpt MSE scale selection requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError(f"expected SM120/SM121, got {capability}")

    source_revision = (source / ".source-revision").read_text().strip()
    if source_revision != args.expected_source_revision:
        raise ValueError(
            f"source revision mismatch: {source_revision} != {args.expected_source_revision}"
        )
    index_path = source / "model.safetensors.index.json"
    source_index = json.loads(index_path.read_text())
    source_weight_map: dict[str, str] = source_index["weight_map"]
    if len(source_weight_map) != EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"unexpected source tensor count: {len(source_weight_map)}")
    if len(set(source_weight_map.values())) != EXPECTED_SOURCE_SHARDS:
        raise ValueError("unexpected source shard count")
    if int(source_index["metadata"]["total_size"]) != EXPECTED_SOURCE_BYTES:
        raise ValueError("unexpected BF16 tensor payload size")

    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise ValueError(f"unexpected architecture: {config.get('architectures')}")
    if config.get("quantization_config"):
        raise ValueError("BF16 source unexpectedly has quantization metadata")
    if (config.get("vision_config") or {}).get("depth") != 24:
        raise ValueError("missing or unexpected GLM vision tower")
    text_config = config.get("text_config") or config
    contract = {
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "moe_intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
        "num_hidden_layers": 45,
        "n_routed_experts": EXPECTED_EXPERTS,
        "num_experts_per_tok": 8,
        "num_nextn_predict_layers": 1,
    }
    for field, expected in contract.items():
        if text_config.get(field) != expected:
            raise ValueError(
                f"unexpected text_config.{field}: {text_config.get(field)!r} != {expected!r}"
            )
    layer_types = text_config.get("layer_types") or []
    if len(layer_types) != 45 or layer_types.count("deepseek_sparse_attention") != 11:
        raise ValueError("unexpected hybrid-attention layer layout")

    expected_expert_names = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in EXPECTED_LAYERS
        for expert in range(EXPECTED_EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    actual_expert_names = {name for name in source_weight_map if ".mlp.experts." in name}
    if actual_expert_names != expected_expert_names:
        missing = sorted(expected_expert_names - actual_expert_names)[:10]
        extra = sorted(actual_expert_names - expected_expert_names)[:10]
        raise ValueError(f"routed expert namespace mismatch: missing={missing} extra={extra}")
    shared_names = [name for name in source_weight_map if ".mlp.shared_experts." in name]
    vision_names = [name for name in source_weight_map if "visual" in name]
    if len(shared_names) != EXPECTED_SHARED_TENSORS:
        raise ValueError(f"unexpected shared-expert tensor count: {len(shared_names)}")
    if len(vision_names) != EXPECTED_VISION_TENSORS:
        raise ValueError(f"unexpected vision tensor count: {len(vision_names)}")

    # Joint gate/up calibration is required by the fused W13 kernel: both halves
    # consume one global alpha. Source sharding is independent of expert
    # boundaries, so record cross-shard pairs and open their partner shard on
    # demand during conversion.
    gate_names = sorted(
        name for name in expected_expert_names if ".gate_proj." in name
    )
    cross_shard_gate_up_pairs = 0
    for gate_name in gate_names:
        up_name = gate_name.replace(".gate_proj.", ".up_proj.")
        if source_weight_map[gate_name] != source_weight_map[up_name]:
            cross_shard_gate_up_pairs += 1

    expert_elements = (
        len(expected_expert_names) * EXPECTED_HIDDEN_SIZE * EXPECTED_INTERMEDIATE_SIZE
    )
    protected_bytes = EXPECTED_SOURCE_BYTES - expert_elements * 2
    expected_output_bytes = (
        protected_bytes
        + expert_elements // 2
        + expert_elements // GROUP_SIZE
        + len(expected_expert_names) * 8
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "source_revision": source_revision,
                    "source_tensors": len(source_weight_map),
                    "source_shards": len(set(source_weight_map.values())),
                    "quantized_source_weights": len(expected_expert_names),
                    "cross_shard_gate_up_pairs": cross_shard_gate_up_pairs,
                    "shared_expert_tensors_protected": len(shared_names),
                    "vision_tensors_protected": len(vision_names),
                    "expected_output_bytes": expected_output_bytes,
                    "expected_output_gib": expected_output_bytes / 1024**3,
                    "toolchain": toolchain,
                    "device": torch.cuda.get_device_name(device),
                    "compute_capability": list(capability),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return

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
            "version": toolchain["modelopt_version"],
            "commit": toolchain["modelopt_commit"],
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
        print(f"[{shard_number}/{len(source_shards)}] {shard_name}", flush=True)
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            for name in sorted(shard_keys):
                if name in processed_source_names:
                    continue
                match = EXPERT_PATTERN.fullmatch(name)
                if match is None:
                    writer.add(name, handle.get_tensor(name), "protected_bf16_or_fp32")
                    processed_source_names.add(name)
                    continue
                projection = match.group(3)
                if projection == "gate_proj":
                    up_name = name.replace(".gate_proj.", ".up_proj.")
                    if source_weight_map[up_name] == shard_name:
                        up_tensor = handle.get_tensor(up_name)
                    else:
                        with safe_open(
                            source / source_weight_map[up_name],
                            framework="pt",
                            device="cpu",
                        ) as up_handle:
                            up_tensor = up_handle.get_tensor(up_name)
                    quantize_gate_up(
                        writer,
                        name,
                        handle.get_tensor(name),
                        up_name,
                        up_tensor,
                        device,
                        metrics,
                    )
                    processed_source_names.update((name, up_name))
                elif projection == "up_proj":
                    # Its gate is the sole owner of joint calibration. If this
                    # shard sorts first, the later gate opens this shard on
                    # demand and marks both source names processed.
                    continue
                else:
                    quantize_down(
                        writer,
                        name,
                        handle.get_tensor(name),
                        device,
                        metrics,
                    )
                    processed_source_names.add(name)
        gc.collect()

    if processed_source_names != set(source_weight_map):
        missing = sorted(set(source_weight_map) - processed_source_names)[:10]
        extra = sorted(processed_source_names - set(source_weight_map))[:10]
        raise ValueError(f"source traversal mismatch: missing={missing} extra={extra}")
    expected_calibration_units = len(EXPECTED_LAYERS) * EXPECTED_EXPERTS * 2
    if len(metrics) != expected_calibration_units:
        raise ValueError(
            f"expected {expected_calibration_units} calibration units, got {len(metrics)}"
        )

    output_index = writer.finish()
    if writer.total_size != expected_output_bytes:
        raise ValueError(
            f"output byte contract mismatch: {writer.total_size} != {expected_output_bytes}"
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
        "purpose": "GLM-5.3-Flash high-quality TP2 E4M3-K32 W4A16 candidate",
        "source": str(source),
        "source_revision": source_revision,
        "source_index_sha256": sha256_file(index_path),
        "output": str(output),
        "toolchain": toolchain,
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
        },
        "recipe": quant_config,
        "selection": {
            "layers": list(EXPECTED_LAYERS),
            "routed_experts_per_layer": EXPECTED_EXPERTS,
            "quantized_source_weights": len(expected_expert_names),
            "cross_shard_gate_up_pairs": cross_shard_gate_up_pairs,
            "calibration_units": len(metrics),
            "native_mtp_layer": 45,
            "native_mtp_routed_experts_quantized": True,
            "shared_expert_tensors_protected": len(shared_names),
            "vision_tensors_protected": len(vision_names),
            "shared_experts_fusion_must_be_disabled": True,
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
        "total_tensor_bytes": writer.total_size,
        "total_tensor_gib": writer.total_size / 1024**3,
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
