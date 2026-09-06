"""Compare actual resolved AIPerf settings, excluding only output directories."""
import copy,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
before=root.parent/'v0.3.1-release-20260906/engine-gate-reviewed'
after=root/'engine-gate'
rows=[]
for path in sorted(after.rglob('profile_export_aiperf.json')):
 relative=path.relative_to(after)
 if relative.parts[0] not in ['decode','prefill']:continue
 b=json.loads((before/relative).read_text());c=json.loads(path.read_text())
 configs=[copy.deepcopy(d['input_config']) for d in [b,c]]
 dirs=[conf['artifacts'].pop('dir') for conf in configs]
 assert configs[0]==configs[1],relative
 assert b['run_info']['random_seed']==c['run_info']['random_seed'],relative
 canonical=json.dumps(configs[0],sort_keys=True,separators=(',',':')).encode()
 rows.append({'cell':str(relative.parent),'workload_sha256':hashlib.sha256(canonical).hexdigest(),'random_seed':c['run_info']['random_seed'],'excluded_artifact_dirs':dirs,'settings_match':True})
assert len(rows)==24
result={'resolved_workloads_match':24,'method':'Compare all exported input_config fields exactly, excluding only artifacts.dir, and compare run_info.random_seed. Saved benchmark-config.yaml templates were separately checked byte-identical.','cells':rows}
(root/'resolved-workload-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print('All 24 resolved workloads and random seeds match; only artifact output directories differ.')
