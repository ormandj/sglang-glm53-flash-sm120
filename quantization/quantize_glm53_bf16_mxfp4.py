#!/usr/bin/env python3
"""Produce and fail-closed validate the GLM-5.3-Flash BF16 -> MXFP4 artifact."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import MXFP4A16
from llmcompressor import model_free_ptq
from safetensors import safe_open


SOURCE_REPOSITORY = "zai-org/GLM-5.3-Flash-BF16"
SOURCE_REVISION = "f12e0fe1f6b2ea274c11a569582edfd99d993c5e"
SOURCE_STORAGE_BYTES = 642_676_400_602
SOURCE_TENSOR_BYTES = 642_646_653_816
SOURCE_FILES = 130
SOURCE_TENSORS = 38_770
QUANTIZED_TENSORS = 37_152  # 43 routed layers * 288 experts * 3 projections
VISION_TENSORS = 347
ROUTED_LAYERS = range(3, 46)  # layer 45 is the native MTP block
EXPERTS_PER_LAYER = 288
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TARGET_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
TARGET_REGEX = (
    r"re:^model\.language_model\.layers\."
    r"(?:[3-9]|[1-3][0-9]|4[0-5])\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).resolve()


SOURCE = env_path("GLM53_BF16_SOURCE", "/scratch/zai-org/GLM-5.3-Flash-BF16")
FINAL = env_path(
    "GLM53_MXFP4_OUTPUT", "/scratch/out/GLM-5.3-Flash-BF16-MXFP4"
)
INCOMPLETE = FINAL.parent / f".{FINAL.name}.incomplete"
MAX_WORKERS = int(os.environ.get("GLM53_QUANT_WORKERS", "12"))
TORCH_THREADS = int(os.environ.get("GLM53_TORCH_THREADS", "4"))


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)


def load_index(root: Path) -> dict:
    index = read_json(root / "model.safetensors.index.json")
    if not isinstance(index.get("weight_map"), dict):
        fail(f"invalid safetensors index under {root}")
    return index


def collect_metadata(root: Path, index: dict) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read every tensor header and require a one-to-one index/shard mapping."""
    by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in index["weight_map"].items():
        by_shard[shard].add(name)

    metadata: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard, indexed_names in sorted(by_shard.items()):
        path = root / shard
        if not path.is_file():
            fail(f"index references missing shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            if actual_names != indexed_names:
                missing = sorted(indexed_names - actual_names)[:5]
                extra = sorted(actual_names - indexed_names)[:5]
                fail(f"index/header disagreement in {shard}: missing={missing}, extra={extra}")
            for name in actual_names:
                tensor_slice = handle.get_slice(name)
                metadata[name] = (
                    tuple(tensor_slice.get_shape()),
                    tensor_slice.get_dtype(),
                )
    if set(metadata) != set(index["weight_map"]):
        fail("metadata traversal did not cover the index exactly")
    return metadata


def load_tensor(root: Path, index: dict, name: str) -> torch.Tensor:
    shard = index["weight_map"][name]
    with safe_open(root / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def expected_targets(names: set[str]) -> set[str]:
    targets: set[str] = set()
    coverage: dict[int, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for name in names:
        match = TARGET_PATTERN.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if layer not in ROUTED_LAYERS:
            fail(f"routed expert tensor appeared outside layers 3-45: {name}")
        if not 0 <= expert < EXPERTS_PER_LAYER:
            fail(f"expert id out of range: {name}")
        targets.add(name)
        coverage[layer][expert].add(projection)

    expected_projection_set = set(PROJECTIONS)
    for layer in ROUTED_LAYERS:
        if set(coverage[layer]) != set(range(EXPERTS_PER_LAYER)):
            fail(f"layer {layer} does not contain exactly experts 0-287")
        for expert in range(EXPERTS_PER_LAYER):
            if coverage[layer][expert] != expected_projection_set:
                fail(
                    f"layer {layer} expert {expert} projection set is "
                    f"{sorted(coverage[layer][expert])}"
                )
    if len(targets) != QUANTIZED_TENSORS:
        fail(f"expected {QUANTIZED_TENSORS} routed weights, found {len(targets)}")
    return targets


def validate_source() -> tuple[dict, dict, set[str]]:
    marker_path = SOURCE / ".download-complete"
    if not marker_path.is_file():
        fail(f"verified source marker is absent: {marker_path}")
    marker = read_json(marker_path)
    expected_marker = {
        "revision": SOURCE_REVISION,
        "files": SOURCE_FILES,
        "bytes": SOURCE_STORAGE_BYTES,
    }
    if marker != expected_marker:
        fail(f"source marker mismatch: {marker!r} != {expected_marker!r}")
    if (SOURCE / ".source-revision").read_text().strip() != SOURCE_REVISION:
        fail("source revision sidecar does not match the pinned BF16 revision")

    config = read_json(SOURCE / "config.json")
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        fail(f"unexpected architecture: {config.get('architectures')}")
    text_config = config.get("text_config") or {}
    if text_config.get("dtype") != "bfloat16":
        fail(f"source text dtype is not bfloat16: {text_config.get('dtype')}")
    if config.get("quantization_config") or text_config.get("quantization_config"):
        fail("source contains a quantization config; refusing double quantization")
    if not config.get("vision_config"):
        fail("source config has no vision_config")
    if not (SOURCE / "processor_config.json").is_file():
        fail("source has no processor_config.json for multimodal preprocessing")

    index = load_index(SOURCE)
    names = set(index["weight_map"])
    if len(names) != SOURCE_TENSORS:
        fail(f"expected {SOURCE_TENSORS} source tensors, found {len(names)}")
    if int(index.get("metadata", {}).get("total_size", -1)) != SOURCE_TENSOR_BYTES:
        fail(f"source tensor byte total is not the pinned {SOURCE_TENSOR_BYTES}")
    if any("weight_scale" in name or "weight_packed" in name for name in names):
        fail("source tensor namespace contains quantization sidecars")
    if len([name for name in names if name.startswith("model.visual.")]) != VISION_TENSORS:
        fail(f"source does not contain exactly {VISION_TENSORS} vision tensors")

    targets = expected_targets(names)
    metadata = collect_metadata(SOURCE, index)
    non_bf16 = [name for name in targets if metadata[name][1] != "BF16"]
    if non_bf16:
        fail(f"routed source weights are not BF16: {non_bf16[:5]}")
    return index, metadata, targets


def build_quant_config() -> tuple[QuantizationConfig, QuantizationScheme]:
    weights = MXFP4A16["weights"].model_copy()
    if not (
        weights.num_bits == 4
        and weights.group_size == 32
        and weights.type == "float"
    ):
        fail(f"MXFP4A16 preset contract changed: {weights}")
    scheme = QuantizationScheme(targets=[TARGET_REGEX], weights=weights)
    return QuantizationConfig(config_groups={"routed_experts": scheme}), scheme


def validate_quant_config(config: dict) -> None:
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        fail("output config has no quantization_config")
    if quant.get("quant_method") != "compressed-tensors":
        fail(f"wrong quant_method: {quant.get('quant_method')}")
    if quant.get("format") != "mxfp4-pack-quantized":
        fail(f"wrong compression format: {quant.get('format')}")
    groups = quant.get("config_groups") or {}
    if set(groups) != {"routed_experts"}:
        fail(f"unexpected quantization groups: {sorted(groups)}")
    group = groups["routed_experts"]
    if group.get("targets") != [TARGET_REGEX]:
        fail(f"output target regex drifted: {group.get('targets')}")
    weights = group.get("weights") or {}
    required = {
        "num_bits": 4,
        "type": "float",
        "strategy": "group",
        "group_size": 32,
        "dynamic": False,
        "symmetric": True,
    }
    for key, value in required.items():
        if weights.get(key) != value:
            fail(f"output MXFP4 weights.{key}={weights.get(key)!r}, expected {value!r}")


def validate_protected_exact(
    source_index: dict,
    output_index: dict,
    protected: set[str],
) -> None:
    started = time.monotonic()
    for ordinal, name in enumerate(sorted(protected), start=1):
        source_tensor = load_tensor(SOURCE, source_index, name)
        output_tensor = load_tensor(INCOMPLETE, output_index, name)
        if not torch.equal(source_tensor, output_tensor):
            fail(f"protected tensor changed: {name}")
        if ordinal % 200 == 0:
            print(
                f"Protected equality: {ordinal}/{len(protected)} tensors",
                flush=True,
            )
    print(
        f"Protected equality: {len(protected)}/{len(protected)} tensors "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )


def validate_roundtrips(
    source_index: dict,
    output_index: dict,
    scheme: QuantizationScheme,
) -> dict:
    probes: list[dict] = []
    for layer in ROUTED_LAYERS:
        expert = (layer * 53) % EXPERTS_PER_LAYER
        for projection in PROJECTIONS:
            module = (
                f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                f"{projection}"
            )
            source_name = module + ".weight"
            packed_name = module + ".weight_packed"
            scale_name = module + ".weight_scale"
            reference = load_tensor(SOURCE, source_index, source_name).float()
            packed = load_tensor(INCOMPLETE, output_index, packed_name)
            scale = load_tensor(INCOMPLETE, output_index, scale_name)
            restored = MXFP4PackedCompressor.decompress(
                {"weight_packed": packed, "weight_scale": scale}, scheme
            )["weight"].float()
            if tuple(restored.shape) != tuple(reference.shape):
                fail(
                    f"round-trip shape mismatch for {module}: "
                    f"{tuple(restored.shape)} != {tuple(reference.shape)}"
                )
            relative_l2 = ((restored - reference).norm() / reference.norm()).item()
            if not math.isfinite(relative_l2):
                fail(f"non-finite round-trip error for {module}")
            probes.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "relative_l2": relative_l2,
                }
            )
    errors = sorted(probe["relative_l2"] for probe in probes)
    p95 = errors[math.ceil(0.95 * len(errors)) - 1]
    summary = {
        "count": len(errors),
        "minimum": min(errors),
        "median": statistics.median(errors),
        "mean": statistics.fmean(errors),
        "p95": p95,
        "maximum": max(errors),
        "probes": probes,
    }
    if summary["minimum"] <= 0.01:
        fail(f"implausibly small MXFP4 round-trip error: {summary['minimum']}")
    if summary["mean"] >= 0.15 or summary["p95"] >= 0.17 or summary["maximum"] >= 0.20:
        fail(f"MXFP4 round-trip error outside audited envelope: {summary}")
    return summary


def validate_output(
    source_index: dict,
    source_metadata: dict,
    targets: set[str],
    scheme: QuantizationScheme,
) -> dict:
    output_config = read_json(INCOMPLETE / "config.json")
    validate_quant_config(output_config)
    if output_config.get("vision_config") is None:
        fail("output config lost vision_config")
    if not (INCOMPLETE / "processor_config.json").is_file():
        fail("output lost processor_config.json")

    output_index = load_index(INCOMPLETE)
    output_metadata = collect_metadata(INCOMPLETE, output_index)
    output_names = set(output_metadata)
    source_names = set(source_metadata)
    protected = source_names - targets
    expected_output_names = set(protected)
    for name in targets:
        module = name.removesuffix(".weight")
        expected_output_names.add(module + ".weight_packed")
        expected_output_names.add(module + ".weight_scale")
    if output_names != expected_output_names:
        missing = sorted(expected_output_names - output_names)[:10]
        extra = sorted(output_names - expected_output_names)[:10]
        fail(f"output namespace mismatch: missing={missing}, extra={extra}")

    packed = [name for name in output_names if name.endswith(".weight_packed")]
    scales = [name for name in output_names if name.endswith(".weight_scale")]
    if len(packed) != QUANTIZED_TENSORS or len(scales) != QUANTIZED_TENSORS:
        fail(f"packed/scale counts are {len(packed)}/{len(scales)}")
    if len(output_names) != SOURCE_TENSORS + QUANTIZED_TENSORS:
        fail(f"unexpected output tensor count: {len(output_names)}")

    for source_name in targets:
        module = source_name.removesuffix(".weight")
        source_shape = source_metadata[source_name][0]
        packed_shape, packed_dtype = output_metadata[module + ".weight_packed"]
        scale_shape, scale_dtype = output_metadata[module + ".weight_scale"]
        if packed_dtype != "U8" or scale_dtype != "U8":
            fail(f"packed dtypes are not U8 for {module}: {packed_dtype}/{scale_dtype}")
        expected_packed_shape = (*source_shape[:-1], source_shape[-1] // 2)
        expected_scale_shape = (*source_shape[:-1], source_shape[-1] // 32)
        if packed_shape != expected_packed_shape or scale_shape != expected_scale_shape:
            fail(
                f"packed shapes wrong for {module}: {packed_shape}/{scale_shape}, "
                f"expected {expected_packed_shape}/{expected_scale_shape}"
            )

    for name in protected:
        if output_metadata[name] != source_metadata[name]:
            fail(
                f"protected tensor metadata changed for {name}: "
                f"{source_metadata[name]} -> {output_metadata[name]}"
            )
    vision_names = [name for name in protected if name.startswith("model.visual.")]
    if len(vision_names) != VISION_TENSORS:
        fail(f"output does not preserve all {VISION_TENSORS} vision tensors")

    validate_protected_exact(source_index, output_index, protected)
    roundtrips = validate_roundtrips(source_index, output_index, scheme)
    tensor_bytes = int(output_index.get("metadata", {}).get("total_size", -1))
    if not 175_000_000_000 <= tensor_bytes <= 195_000_000_000:
        fail(f"output tensor bytes outside expected MXFP4 envelope: {tensor_bytes}")
    return {
        "source_tensors": len(source_names),
        "output_tensors": len(output_names),
        "quantized_tensors": len(packed),
        "protected_tensors_exact": len(protected),
        "vision_tensors_exact": len(vision_names),
        "tensor_bytes": tensor_bytes,
        "roundtrip": roundtrips,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_file_manifest() -> dict:
    excluded = {".artifact-manifest.json", ".quant-complete"}
    files = []
    for path in sorted(candidate for candidate in INCOMPLETE.rglob("*") if candidate.is_file()):
        relative = path.relative_to(INCOMPLETE).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if len(files) % 20 == 0:
            print(f"Artifact hashes: {len(files)} files", flush=True)
    payload = {
        "algorithm": "sha256",
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    write_json_atomic(INCOMPLETE / ".artifact-manifest.json", payload)
    return payload


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def installed_packages() -> dict[str, str]:
    return dict(
        sorted(
            (
                distribution.metadata["Name"].lower(),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        )
    )


def main() -> None:
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "final": str(FINAL),
                "incomplete": str(INCOMPLETE),
                "workers": MAX_WORKERS,
                "torch_threads": TORCH_THREADS,
            },
            indent=2,
        ),
        flush=True,
    )
    if FINAL.exists():
        fail(f"final artifact already exists; refusing overwrite: {FINAL}")
    if INCOMPLETE.exists():
        fail(f"incomplete artifact already exists; inspect it before removal: {INCOMPLETE}")
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)

    source_index, source_metadata, targets = validate_source()
    quant_config, scheme = build_quant_config()
    started = time.time()
    print(
        f"Source validated; quantizing exactly {len(targets)} BF16 routed weights",
        flush=True,
    )
    model_free_ptq(
        model_stub=SOURCE,
        save_directory=INCOMPLETE,
        config=quant_config,
        max_workers=MAX_WORKERS,
        device="cpu",
    )
    for copied_marker in (".download-complete", ".source-revision"):
        path = INCOMPLETE / copied_marker
        if path.exists():
            path.unlink()

    validation = validate_output(source_index, source_metadata, targets, scheme)
    provenance = {
        "schema": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "storage_bytes": SOURCE_STORAGE_BYTES,
            "tensor_bytes": SOURCE_TENSOR_BYTES,
            "format": "bfloat16",
        },
        "quantization": {
            "format": "mxfp4-pack-quantized",
            "scheme": "MXFP4A16",
            "bits_per_weight": 4.25,
            "group_size": 32,
            "scope": "routed experts in layers 3-45; layer 45 is native MTP",
            "target_regex": TARGET_REGEX,
        },
        "toolchain": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "compressed-tensors": package_version("compressed-tensors"),
            "llmcompressor": package_version("llmcompressor"),
            "safetensors": package_version("safetensors"),
            "compressed_tensors_commit": os.environ.get(
                "COMPRESSED_TENSORS_COMMIT", "unknown"
            ),
            "compressed_tensors_tree": os.environ.get(
                "COMPRESSED_TENSORS_TREE", "unknown"
            ),
            "llm_compressor_commit": os.environ.get(
                "LLM_COMPRESSOR_COMMIT", "unknown"
            ),
            "llm_compressor_tree": os.environ.get(
                "LLM_COMPRESSOR_TREE", "unknown"
            ),
            "installed_packages": installed_packages(),
        },
        "validation": validation,
        "wall_seconds_before_hashing": time.time() - started,
    }
    write_json_atomic(INCOMPLETE / "quantization-provenance.json", provenance)
    manifest = write_file_manifest()
    manifest_sha256 = sha256_file(INCOMPLETE / ".artifact-manifest.json")
    completion = {
        "schema": 1,
        "source_revision": SOURCE_REVISION,
        "format": "mxfp4-pack-quantized",
        "quantized_tensors": QUANTIZED_TENSORS,
        "vision_enabled": True,
        "native_mtp_quantized": True,
        "artifact_files": len(manifest["files"]),
        "artifact_bytes": manifest["total_bytes"],
        "artifact_manifest_sha256": manifest_sha256,
        "completed_unix": int(time.time()),
    }
    write_json_atomic(INCOMPLETE / ".quant-complete", completion)
    os.sync()
    INCOMPLETE.rename(FINAL)
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    print(f"Published validated artifact atomically at {FINAL}", flush=True)


if __name__ == "__main__":
    main()
