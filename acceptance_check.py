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
for dist, name in [('nvidia-modelopt', 'modelopt'), ('sgl-deep-gemm', 'deepgemm'), ('transformers', 'transformers')]:
    assert md.version(dist) == lock['integration'][name]['package_version']
transformers_url = json.loads(md.distribution('transformers').read_text('direct_url.json'))['url']
assert transformers_url == 'file:///opt/glm53/sources/transformers'
for dist, version in lock['processor_dependencies'].items():
    assert md.version(dist) == version, dist
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

# AutoProcessor can silently return a text tokenizer when a model's processor
# class is unavailable. Verify actual pixels and expansion, not just imports.
from hashlib import sha256
from huggingface_hub import snapshot_download
from PIL import Image
from sglang.srt.utils.hf_transformers.processor import get_processor
from transformers import Glm5NextProcessor

metadata_files = ['config.json', 'processor_config.json', 'tokenizer_config.json',
                  'tokenizer.json', 'chat_template.jinja']
model_metadata = Path(snapshot_download(
    lock['model']['source_repository'], revision=lock['model']['source_revision'],
    allow_patterns=metadata_files,
))
processor = get_processor(str(model_metadata), image_processor_backend='torchvision',
                          trust_remote_code=False, local_files_only=True)
assert isinstance(processor, Glm5NextProcessor), type(processor).__name__
tokens = processor(
    text=['<|begin_of_image|><|image|><|end_of_image|>'],
    images=[Image.new('RGB', (224, 224), 'red')], device='cpu', return_tensors='pt',
)
assert tokens['pixel_values'].numel() > 0
grid = tokens['image_grid_thw']
assert tuple(grid.shape) == (1, 3)
expected = int(grid[0].prod()) // processor.image_processor.merge_size ** 2
actual = int((tokens['input_ids'] == processor.image_token_id).sum())
assert actual == expected and actual > 1, (actual, expected)
assert tokens['pixel_values'].device.type == 'cpu'
(root / 'provenance/vision-processor-acceptance.json').write_text(json.dumps({
    'model_revision': lock['model']['source_revision'],
    'transformers_revision': lock['integration']['transformers']['head'],
    'processor': type(processor).__name__, 'image_tokens': actual,
    'pixel_values_shape': list(tokens['pixel_values'].shape),
    'metadata_sha256': {name: sha256((model_metadata / name).read_bytes()).hexdigest()
                        for name in metadata_files},
}, indent=2) + '\n')
print('Exact source packages, native draft, GLM sparse MLA and real vision processor contracts verified; GPU qualification required.')
