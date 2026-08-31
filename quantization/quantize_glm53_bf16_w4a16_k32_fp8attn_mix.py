#!/usr/bin/env python3
"""Produce the GLM-5.3-Flash MIXED_PRECISION candidate: W4A16-K32 experts
plus FP8 [128,128]-block weight-only attention and shared experts.

The routed-expert path is byte-identical to the audited
``quantize_glm53_bf16_w4a16_e4m3_k32.py`` recipe (joint gate/up MSE sweep,
E2M1 + E4M3-K32 + FP32 global scales). New in this candidate:

- every eligible 2D BF16 ``self_attn`` projection (KDA q/k/v/b/f/g/o and
  MLA q_a/q_b/kv_a/kv_b/o) and every shared-expert projection is stored as
  FP8 E4M3 with a float32 ``weight_scale_inv`` per [128,128] block, the
  serialized layout SGLang's ``Fp8LinearMethod`` block path loads
  (``FP8_PB_WO`` in ``ModelOptMixedPrecisionConfig``);
- eligibility is strict: 2D, BF16, and both dimensions divisible by 256 so
  every TP2 shard keeps whole scale blocks. Ineligible tensors stay
  byte-identical BF16 and resolve to UnquantizedLinearMethod.
- the DSA ``indexer.*`` tensors always stay BF16: the indexer head-gate
  reads the raw ``weights_proj`` weight, and the whole submodule is small.
- embeddings, LM head, vision tower, routers, norms, convs, ``eh_proj``,
  and ``shared_head`` stay byte-identical BF16.

The metadata is ``quant_method: modelopt_mixed`` / ``quant_algo:
MIXED_PRECISION`` with a ``quantized_layers`` map and a
``packed_modules_mapping`` for SGLang's fused shards. Output is published
atomically only after source-contract, reconstruction, round-trip, and byte
checks pass.
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
FP8_BLOCK = 128
FP8_MAX = 448.0
EXPECTED_MODELOPT_COMMIT = "022767c7ab3d7d36211affd85e5c496770cde768"
EXPECTED_SOURCE_REVISION = "f12e0fe1f6b2ea274c11a569582edfd99d993c5e"
EXPECTED_SOURCE_BYTES = 642_646_653_816
EXPECTED_SOURCE_TENSORS = 38_770
EXPECTED_SOURCE_SHARDS = 120
EXPECTED_LAYERS = tuple(range(3, 46))
EXPECTED_EXPERTS = 288
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_INTERMEDIATE_SIZE = 2048

EXPERT_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
# Layer 45 is the NextN/MTP draft layer; its weights load through the
# draft model's remapped namespace (model.layers.45 -> model.decoder) where
# mixed-config resolution is unproven, and a silently unquantized draft
# collapses speculative acceptance to zero. Its attention and shared
# experts stay BF16; only layers 0-44 are FP8 candidates.
FP8_CANDIDATE_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?:[0-9]|[1-3][0-9]|4[0-4])\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj|f_b_proj"
    r"|g_a_proj|g_b_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa"
    r"|kv_b_proj)"
    r"|mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj))\.weight$"
)

# Modules the mixed loader must route to UnquantizedLinearMethod even though
# they sit next to quantized ones. Anything absent from quantized_layers is
# unquantized by default; this list is the explicit, reviewable statement.
IGNORE_MODULES = [
    "lm_head",
    "model.embed_tokens",
    "model.language_model.embed_tokens",
    "*.embed_tokens",
    "*.self_attn.indexer.*",
    "*.mlp.gate",
    "*.mlp.gate_up_proj",
    "*.mlp.gate_proj",
    "*.mlp.up_proj",
    "*.mlp.down_proj",
    "*.eh_proj",
    "*.shared_head.*",
    "model.visual.*",
    "visual.*",
    "*.visual.*",
]

# SGLang fuses these shards at load; every member of a fused shard must carry
# the same quant_algo for _resolve_quant_algo to accept it.
PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-source-revision", default=EXPECTED_SOURCE_REVISION)
    parser.add_argument("--max-shard-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--max-aggregate-relative-l2", type=float, default=0.15)
    parser.add_argument("--min-matrix-cosine", type=float, default=0.95)
    parser.add_argument("--max-fp8-matrix-relative-l2", type=float, default=0.04)
    parser.add_argument("--min-fp8-matrix-cosine", type=float, default=0.999)
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
    if identities["modelopt_commit"] != EXPECTED_MODELOPT_COMMIT:
        raise RuntimeError(
            f"modelopt commit mismatch: {identities['modelopt_commit']!r} != "
            f"{EXPECTED_MODELOPT_COMMIT!r}"
        )
    for key, value in identities.items():
        if not value:
            raise RuntimeError(f"missing toolchain identity: {key}")
    return identities


def quantize_matrix(
    weight_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Audited W4A16 E2M1/E4M3-K32 MSE recipe; byte-identical to rc.19."""
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


def fp8_eligible(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    if FP8_CANDIDATE_PATTERN.fullmatch(name) is None:
        return False
    if len(shape) != 2 or dtype != torch.bfloat16:
        return False
    # Both dimensions must keep whole [128,128] scale blocks on every TP2
    # shard, whichever axis the serving layer shards.
    return shape[0] % (2 * FP8_BLOCK) == 0 and shape[1] % (2 * FP8_BLOCK) == 0


def quantize_fp8_block(
    weight_cpu: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """FP8 E4M3 weights with float32 per-[128,128]-block inverse scales."""
    out_dim, in_dim = weight_cpu.shape
    weight = weight_cpu.to(device=device, dtype=torch.float32).contiguous()
    blocked = weight.view(
        out_dim // FP8_BLOCK, FP8_BLOCK, in_dim // FP8_BLOCK, FP8_BLOCK
    )
    amax = blocked.abs().amax(dim=(1, 3))
    if not torch.isfinite(amax).all():
        raise ValueError("non-finite block amax")
    scale = amax / FP8_MAX
    scale[scale == 0] = 1.0
    scaled = blocked / scale[:, None, :, None]
    quant = scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    dequant = quant.to(torch.float32) * scale[:, None, :, None]
    error = dequant - blocked
    error_sq = error.square().sum(dtype=torch.float64).item()
    reference_sq = blocked.square().sum(dtype=torch.float64).item()
    metrics = {
        "error_sq": error_sq,
        "reference_sq": reference_sq,
        "relative_l2": math.sqrt(error_sq / reference_sq) if reference_sq else 0.0,
        "cosine": torch.nn.functional.cosine_similarity(
            dequant.reshape(1, -1), blocked.reshape(1, -1), dim=1
        ).item(),
        "max_abs_error": error.abs().amax().item(),
    }
    packed = quant.view(out_dim, in_dim).detach().cpu().contiguous()
    scale_cpu = scale.detach().cpu().contiguous()
    del weight, blocked, amax, scale, scaled, quant, dequant, error
    return packed, scale_cpu, metrics


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


def add_expert_projection(
    writer: ShardWriter,
    base: str,
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    writer.add(f"{base}.weight", packed, "routed_expert_packed")
    writer.add(f"{base}.weight_scale", scale, "routed_expert_e4m3_scale")
    writer.add(f"{base}.weight_scale_2", global_scale.clone(), "routed_expert_global_scale")
    writer.add(
        f"{base}.input_scale",
        torch.tensor(1.0, dtype=torch.float32),
        "routed_expert_input_scale",
    )


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

    expected_expert_names = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in EXPECTED_LAYERS
        for expert in range(EXPECTED_EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    actual_expert_names = {name for name in source_weight_map if ".mlp.experts." in name}
    if actual_expert_names != expected_expert_names:
        raise ValueError("routed expert namespace mismatch")

    # Determine FP8 eligibility from shard headers before conversion so the
    # byte contract, quantized_layers map, and preflight all agree.
    import struct

    tensor_meta: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard_name in sorted(set(source_weight_map.values())):
        with (source / shard_name).open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in tensor_meta:
                raise ValueError(f"tensor name in multiple shards: {name}")
            if source_weight_map.get(name) != shard_name:
                raise ValueError(
                    f"index maps {name} to {source_weight_map.get(name)!r}, "
                    f"found in {shard_name}"
                )
            tensor_meta[name] = (tuple(meta["shape"]), meta["dtype"])
    if set(tensor_meta) != set(source_weight_map):
        raise ValueError("shard headers and index disagree on tensor namespace")

    fp8_names: list[str] = []
    fp8_rejected: list[str] = []
    for name, (shape, dtype_str) in sorted(tensor_meta.items()):
        if FP8_CANDIDATE_PATTERN.fullmatch(name) is None:
            continue
        dtype = torch.bfloat16 if dtype_str == "BF16" else None
        if dtype is not None and fp8_eligible(name, shape, dtype):
            fp8_names.append(name)
        else:
            fp8_rejected.append(name)

    # SGLang loads some shards fused (fused_qkv_a_proj_with_mqa, gate_up_proj)
    # and rejects a fused layer whose members carry different quant_algos, so a
    # fused group is quantized all-or-nothing.
    fused_groups = (
        (
            ".self_attn.q_a_proj.weight",
            ".self_attn.kv_a_proj_with_mqa.weight",
        ),
        (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
        ),
        (
            ".shared_experts.gate_proj.weight",
            ".shared_experts.up_proj.weight",
        ),
    )
    fp8_name_lookup = set(fp8_names)
    for group in fused_groups:
        for name in list(fp8_name_lookup):
            for suffix in group:
                if name.endswith(suffix):
                    prefix = name.removesuffix(suffix)
                    members = [prefix + other for other in group]
                    # All-or-nothing: if any member of a fused shard is
                    # missing from the source or ineligible, drop them all.
                    if not all(m in fp8_name_lookup for m in members):
                        for m in members:
                            if m in fp8_name_lookup:
                                fp8_name_lookup.discard(m)
                                fp8_rejected.append(
                                    m + " (fused-group member ineligible)"
                                )
                    break
    fp8_names = sorted(fp8_name_lookup)

    expert_elements = (
        len(expected_expert_names) * EXPECTED_HIDDEN_SIZE * EXPECTED_INTERMEDIATE_SIZE
    )
    fp8_elements = sum(
        tensor_meta[name][0][0] * tensor_meta[name][0][1] for name in fp8_names
    )
    fp8_scale_elements = sum(
        (tensor_meta[name][0][0] // FP8_BLOCK) * (tensor_meta[name][0][1] // FP8_BLOCK)
        for name in fp8_names
    )
    protected_bytes = (
        EXPECTED_SOURCE_BYTES - expert_elements * 2 - fp8_elements * 2
    )
    expected_output_bytes = (
        protected_bytes
        + expert_elements // 2
        + expert_elements // GROUP_SIZE
        + len(expected_expert_names) * 8
        + fp8_elements
        + fp8_scale_elements * 4
    )

    # SGLang builds GLM runtime modules under model.layers.* (the loader
    # strips "language_model."); emit both namespaces so resolution works
    # regardless of which prefix the serving stack presents.
    def _dual_namespace(base: str) -> tuple[str, str]:
        stripped = base.replace("model.language_model.", "model.", 1)
        return (base, stripped)

    quantized_layers: dict[str, dict[str, Any]] = {}
    for layer in EXPECTED_LAYERS:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for key in _dual_namespace(
                f"model.language_model.layers.{layer}.mlp.experts.0.{projection}"
            ):
                quantized_layers[key] = {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": GROUP_SIZE,
                }
    for name in fp8_names:
        for key in _dual_namespace(name.removesuffix(".weight")):
            quantized_layers[key] = {"quant_algo": "FP8_PB_WO"}

    quant_config = {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_scheme": None,
        "quant_method": "modelopt_mixed",
        "ignore": IGNORE_MODULES,
        "packed_modules_mapping": PACKED_MODULES_MAPPING,
        "quantized_layers": quantized_layers,
        "producer": {
            "name": "modelopt",
            "version": toolchain["modelopt_version"],
            "commit": toolchain["modelopt_commit"],
        },
    }

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "source_revision": source_revision,
                    "quantized_expert_weights": len(expected_expert_names),
                    "fp8_block_weights": len(fp8_names),
                    "fp8_rejected_candidates": fp8_rejected,
                    "expected_output_bytes": expected_output_bytes,
                    "expected_output_gib": expected_output_bytes / 1024**3,
                    "quantized_layers_entries": len(quantized_layers),
                    "toolchain": toolchain,
                    "device": torch.cuda.get_device_name(device),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return

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
    expert_metrics: list[dict[str, Any]] = []
    fp8_metrics: list[dict[str, Any]] = []
    fp8_name_set = set(fp8_names)
    processed: set[str] = set()
    source_shards = sorted(set(source_weight_map.values()))
    for shard_number, shard_name in enumerate(source_shards, 1):
        print(f"[{shard_number}/{len(source_shards)}] {shard_name}", flush=True)
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                if name in processed:
                    continue
                match = EXPERT_PATTERN.fullmatch(name)
                if match is not None:
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
                        gate = handle.get_tensor(name)
                        expected_shape = (
                            EXPECTED_INTERMEDIATE_SIZE,
                            EXPECTED_HIDDEN_SIZE,
                        )
                        if (
                            tuple(gate.shape) != expected_shape
                            or tuple(up_tensor.shape) != expected_shape
                        ):
                            raise ValueError(f"unexpected gate/up shape at {name}")
                        joint = torch.cat((gate, up_tensor), dim=0)
                        packed, scale, global_scale, metric = quantize_matrix(
                            joint, device
                        )
                        gate_packed, up_packed = packed.split(
                            EXPECTED_INTERMEDIATE_SIZE, dim=0
                        )
                        gate_scale, up_scale = scale.split(
                            EXPECTED_INTERMEDIATE_SIZE, dim=0
                        )
                        add_expert_projection(
                            writer,
                            name.removesuffix(".weight"),
                            gate_packed,
                            gate_scale,
                            global_scale,
                        )
                        add_expert_projection(
                            writer,
                            up_name.removesuffix(".weight"),
                            up_packed,
                            up_scale,
                            global_scale,
                        )
                        expert_metrics.append(
                            {"source": [name, up_name], "unit": "joint_gate_up", **metric}
                        )
                        processed.update((name, up_name))
                        del joint, packed, scale, global_scale, gate, up_tensor
                        torch.cuda.empty_cache()
                    elif projection == "up_proj":
                        continue
                    else:
                        tensor = handle.get_tensor(name)
                        if tuple(tensor.shape) != (
                            EXPECTED_HIDDEN_SIZE,
                            EXPECTED_INTERMEDIATE_SIZE,
                        ):
                            raise ValueError(f"unexpected down shape at {name}")
                        packed, scale, global_scale, metric = quantize_matrix(
                            tensor, device
                        )
                        add_expert_projection(
                            writer,
                            name.removesuffix(".weight"),
                            packed,
                            scale,
                            global_scale,
                        )
                        expert_metrics.append(
                            {"source": [name], "unit": "down", **metric}
                        )
                        processed.add(name)
                        del packed, scale, global_scale, tensor
                        torch.cuda.empty_cache()
                    continue
                if name in fp8_name_set:
                    tensor = handle.get_tensor(name)
                    packed, scale_inv, metric = quantize_fp8_block(tensor, device)
                    base = name.removesuffix(".weight")
                    writer.add(f"{base}.weight", packed, "fp8_block_weight")
                    writer.add(
                        f"{base}.weight_scale_inv", scale_inv, "fp8_block_scale_inv"
                    )
                    if metric["relative_l2"] > args.max_fp8_matrix_relative_l2:
                        raise ValueError(
                            f"FP8 relative L2 {metric['relative_l2']} exceeds "
                            f"{args.max_fp8_matrix_relative_l2} at {name}"
                        )
                    if metric["cosine"] < args.min_fp8_matrix_cosine:
                        raise ValueError(
                            f"FP8 cosine {metric['cosine']} below "
                            f"{args.min_fp8_matrix_cosine} at {name}"
                        )
                    fp8_metrics.append({"source": name, **metric})
                    processed.add(name)
                    del packed, scale_inv, tensor
                    torch.cuda.empty_cache()
                    continue
                writer.add(name, handle.get_tensor(name), "protected_bf16_or_fp32")
                processed.add(name)
        gc.collect()

    if processed != set(source_weight_map):
        missing = sorted(set(source_weight_map) - processed)[:10]
        raise ValueError(f"source traversal mismatch: missing={missing}")
    expected_units = len(EXPECTED_LAYERS) * EXPECTED_EXPERTS * 2
    if len(expert_metrics) != expected_units:
        raise ValueError(
            f"expected {expected_units} expert units, got {len(expert_metrics)}"
        )
    if len(fp8_metrics) != len(fp8_names):
        raise ValueError(
            f"expected {len(fp8_names)} fp8 units, got {len(fp8_metrics)}"
        )

    output_index = writer.finish()
    if writer.total_size != expected_output_bytes:
        raise ValueError(
            f"output byte contract mismatch: {writer.total_size} != {expected_output_bytes}"
        )

    error_sq = sum(row["error_sq"] for row in expert_metrics)
    reference_sq = sum(row["reference_sq"] for row in expert_metrics)
    aggregate_relative_l2 = math.sqrt(error_sq / reference_sq)
    min_cosine = min(row["cosine"] for row in expert_metrics)
    if aggregate_relative_l2 > args.max_aggregate_relative_l2:
        raise ValueError(f"expert aggregate relative L2 {aggregate_relative_l2} too high")
    if min_cosine < args.min_matrix_cosine:
        raise ValueError(f"expert minimum cosine {min_cosine} too low")
    fp8_error_sq = sum(row["error_sq"] for row in fp8_metrics)
    fp8_reference_sq = sum(row["reference_sq"] for row in fp8_metrics)
    fp8_aggregate = math.sqrt(fp8_error_sq / fp8_reference_sq)

    # Output round trip: every tensor re-read, shapes/dtypes/categories match,
    # protected hashes intact, all scales finite.
    expected_names = set(writer.tensor_specs)
    if set(output_index["weight_map"]) != expected_names:
        raise ValueError("output index tensor namespace mismatch")
    seen: set[str] = set()
    roundtrip_counts: dict[str, int] = defaultdict(int)
    for shard_path in sorted(incomplete.glob("model-*-of-*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise ValueError(f"duplicate output tensor: {name}")
                seen.add(name)
                tensor = handle.get_tensor(name)
                shape, dtype, category, expected_digest = writer.tensor_specs[name]
                if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(f"round-trip mismatch for {name}")
                if expected_digest is not None and sha256_tensor(tensor) != expected_digest:
                    raise ValueError(f"protected tensor changed: {name}")
                if "scale" in category and not torch.isfinite(tensor.float()).all():
                    raise ValueError(f"non-finite scale: {name}")
                roundtrip_counts[category] += 1
    if seen != expected_names:
        raise ValueError("output traversal namespace mismatch")

    config["quantization_config"] = quant_config
    json_dump(incomplete / "config.json", config)
    summary = {
        "schema": 2,
        "purpose": (
            "GLM-5.3-Flash MIXED_PRECISION candidate: W4A16-K32 experts + "
            "FP8 [128,128]-block weight-only attention and shared experts"
        ),
        "source": str(source),
        "source_revision": source_revision,
        "source_index_sha256": sha256_file(index_path),
        "output": str(output),
        "toolchain": toolchain,
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
        },
        "recipe_note": (
            "expert recipe byte-identical to quantize_glm53_bf16_w4a16_e4m3_k32.py"
        ),
        "selection": {
            "quantized_expert_weights": len(expected_expert_names),
            "fp8_block_weights": len(fp8_names),
            "fp8_rejected_candidates": fp8_rejected,
            "indexer_kept_bf16": True,
            "vision_embeddings_head_kept_bf16": True,
            "shared_experts_fusion_must_be_disabled": True,
        },
        "reconstruction": {
            "expert_aggregate_relative_l2": aggregate_relative_l2,
            "expert_min_cosine": min_cosine,
            "fp8_aggregate_relative_l2": fp8_aggregate,
            "fp8_max_relative_l2": max(row["relative_l2"] for row in fp8_metrics),
            "fp8_min_cosine": min(row["cosine"] for row in fp8_metrics),
        },
        "roundtrip_tensor_counts": dict(sorted(roundtrip_counts.items())),
        "tensor_bytes": dict(sorted(writer.category_bytes.items())),
        "total_tensor_bytes": writer.total_size,
        "total_tensor_gib": writer.total_size / 1024**3,
        "elapsed_seconds": time.time() - started,
    }
    json_dump(incomplete / "quantization-manifest.json", summary)
    json_dump(incomplete / "reconstruction-metrics-experts.json", expert_metrics)
    json_dump(incomplete / "reconstruction-metrics-fp8.json", fp8_metrics)
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
