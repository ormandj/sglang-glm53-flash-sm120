"""Reproduce exact upstream trees before applying explicit packaging overrides."""
import hashlib
import json
import pathlib
import subprocess

root = pathlib.Path('/opt/glm53')
lock = json.loads((root / 'stack.lock.json').read_text())
receipt = {}
for name, source in lock['integration'].items():
    path = root / 'sources' / name
    path.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    git('init', '-q')
    git('remote', 'add', 'origin', source['repository'])
    git('fetch', '--depth=1', 'origin', source['head'])
    git('checkout', '--detach', 'FETCH_HEAD')
    assert git('rev-parse', 'HEAD') == source['head']
    assert git('rev-parse', 'HEAD^{tree}') == source.get('upstream_tree', source['tree'])
    if 'patch' in source:
        patch = root / source['patch']
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == source['patch_sha256']
        git('apply', '--index', str(patch))
        assert git('write-tree') == source['tree']
    git('submodule', 'update', '--init', '--recursive', '--depth=1')
    receipt[name] = dict(source, submodules=git('submodule', 'status', '--recursive'))
# These change package metadata, not serving code. Retain the exact diff separately.
p = root / 'sources/sglang/python/pyproject.toml'
s = p.read_text()
for old, new in [('flashinfer_python[cu13]==0.6.18', 'flashinfer_python[cu13]==0.7.0'), ('nvidia-cutlass-dsl[cu13]==4.6.2', 'nvidia-cutlass-dsl[cu13]==4.8.0'), ('quack-kernels==0.6.4', 'quack-kernels==0.6.5')]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
p.write_text(s)
(root / 'sources/deepgemm/sgl_deep_gemm/VERSION').write_text(lock['integration']['deepgemm']['package_version'] + '\n')
(root / 'provenance').mkdir(exist_ok=True)
for name in lock['integration']:
    diff = subprocess.check_output(['git', '-C', str(root / 'sources' / name), 'diff', '--binary'])
    (root / 'provenance' / f'{name}-packaging.patch').write_bytes(diff)
    receipt[name]['packaging_diff_sha256'] = hashlib.sha256(diff).hexdigest()
(root / 'provenance/source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
