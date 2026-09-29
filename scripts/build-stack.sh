#!/usr/bin/env bash
set -euo pipefail
root=/opt/glm53
py=/opt/sglang/bin/python
rustup toolchain install 1.92 --profile minimal
uv pip install --python "$py" 'setuptools>=80' setuptools-rust 'setuptools-scm>=8,<10' wheel build scikit-build-core cmake ninja
uv run --no-project --python "$py" python "$root/scripts/prepare-stack.py"
uv pip uninstall --python "$py" flashinfer-python flashinfer-cubin flashinfer-jit-cache sgl-deep-gemm deep-gemm || true
cd "$root/sources/flashinfer"
uv pip install --python "$py" --no-build-isolation '.[cu13]' 'nvidia-cutlass-dsl[cu13]==4.8.0'
artifact=$(uv run --no-project --python "$py" python -c 'import json; a=json.load(open("/opt/glm53/stack.lock.json"))["artifacts"]["flashinfer_cubin"]; print(a["url"]+"#sha256="+a["sha256"])')
uv pip install --python "$py" --no-deps "flashinfer-cubin @ $artifact"
cd "$root/sources/deepgemm"
uv run --no-project --python "$py" bash build_sgl_deep_gemm.sh
uv pip install --python "$py" --no-deps dist/*.whl
sha256sum dist/*.whl > "$root/provenance/deepgemm-wheels.sha256"
cd "$root/sources/modelopt"
version=$(uv run --no-project --python "$py" python -c 'import json; print(json.load(open("/opt/glm53/stack.lock.json"))["integration"]["modelopt"]["package_version"])')
SETUPTOOLS_SCM_PRETEND_VERSION="$version" uv pip install --python "$py" --no-deps .
uv pip install --python "$py" --no-build-isolation "$root/sources/transformers" \
  tokenizers==0.23.1 huggingface-hub==1.31.0 safetensors==0.8.0
cd "$root/sources/sglang"
uv pip install --python "$py" --no-build-isolation -e ./python typeguard==4.4.4 'pillow>=12.3.0' protobuf==6.33.5 grpcio-tools==1.81.1 accelerate==1.12.0
# Build the AOT package from the same patched tree as the Python/JIT runtime.
# CUDA 13 upstream CMake includes SM120; omit unrelated pre-SM90 and FA3 targets.
CMAKE_BUILD_PARALLEL_LEVEL=8 uv build --python "$py" --wheel --no-build-isolation \
  --out-dir "$root/wheels/sglang-kernel" ./python/sglang/kernels/aot \
  -Ccmake.define.ENABLE_BELOW_SM90=OFF \
  -Ccmake.define.SGL_KERNEL_ENABLE_FA3=OFF \
  -Ccmake.define.SGL_KERNEL_COMPILE_THREADS=1
sha256sum "$root"/wheels/sglang-kernel/*.whl > "$root/provenance/sglang-kernel-wheels.sha256"
uv pip install --python "$py" --no-deps --reinstall "$root"/wheels/sglang-kernel/*.whl
uv pip uninstall --python "$py" moviepy
uv pip install --python "$py" --no-deps nvidia-nccl-cu13==2.30.7
uv pip freeze --python "$py" > "$root/provenance/installed-packages.txt"
CUDA_VISIBLE_DEVICES=9 uv run --no-project --python "$py" python "$root/acceptance_check.py"
