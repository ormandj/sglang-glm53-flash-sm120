"""Build-time source, package and CPU contract checks; GPU serving is qualified separately."""
import ast
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

root = Path('/opt/glm53')
lock = json.loads((root / 'stack.lock.json').read_text())
import sglang
import flashinfer

assert Path(inspect.getfile(sglang)).is_relative_to(root / 'sources/sglang/python')
assert flashinfer.__version__ == lock['integration']['flashinfer']['package_version']
assert flashinfer.__git_commit__ == lock['integration']['flashinfer']['head']
for dist, name in [('nvidia-modelopt', 'modelopt'), ('sgl-deep-gemm', 'deepgemm')]:
    assert md.version(dist) == lock['integration'][name]['package_version']
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:2:16:8'
assert md.version('nvidia-cutlass-dsl') == '4.8.0'
assert md.version('quack-kernels') == '0.6.5'
# A version match alone could still select the unpatched base-image wheel.
kernel = md.distribution('sglang-kernel')
kernel_url = json.loads(kernel.read_text('direct_url.json'))['url']
assert kernel.version == '0.4.7'
assert kernel_url.startswith('file:///opt/glm53/wheels/sglang-kernel/')
assert (root / 'provenance/sglang-kernel-wheels.sha256').is_file()

# Parse every changed Python file so unused alternate paths cannot hide syntax errors.
import subprocess
changed = subprocess.check_output(['git', '-C', str(root / 'sources/sglang'), 'diff', '--cached', '--name-only'], text=True).splitlines()
for name in changed:
    path = root / 'sources/sglang' / name
    if path.suffix == '.py' and path.exists():
        ast.parse(path.read_text(), filename=name)
from flashinfer.mla import SparseMLASm120Wrapper, supported_sparse_mla_sm120_configs
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import prepare_w4a16_modelopt_e4m3_k32_weights
assert supported_sparse_mla_sm120_configs()['glm53_nope'].bytes_per_token == 528
assert callable(SparseMLASm120Wrapper.run)
assert callable(prepare_w4a16_modelopt_e4m3_k32_weights)
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.srt.models.deepseek_nextn import DeepseekModelNextN
q = type('Q', (), {'get_name': lambda self: 'modelopt_fp4'})()
assert DeepseekModelNextN._resolve_modelopt_fp4_quant_config(q, False) is None
assert DeepseekModelNextN._resolve_modelopt_fp4_quant_config(q, True) is q
g = Glm5NextForConditionalGenerationNextN.__new__(Glm5NextForConditionalGenerationNextN)
assert g._resolve_nextn_quant_config(SimpleNamespace(num_hidden_layers=45, quantization_config={'ignore':['model.layers.45.*']}), q) is None
assert g._resolve_nextn_quant_config(SimpleNamespace(num_hidden_layers=45, quantization_config={'ignore':['*.self_attn.*']}), q) is q
from sglang.srt.arg_groups import model_hook
from sglang.srt.environ import envs
model_hook.is_sm120_supported = lambda: True
model_hook._apply_glm5_next_sm120_defaults('Glm5NextForConditionalGeneration')
assert not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
from sglang.kernels.ops.attention.flash_mla_sm120 import _GLM53_NOPE_FLASHINFER_TOPK
assert _GLM53_NOPE_FLASHINFER_TOPK == 2176
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import get_mla_host_kv_cache_dim
assert callable(get_mla_host_kv_cache_dim)
print('Exact source packages, native draft resolution and GLM sparse MLA contracts verified; GPU qualification required.')
