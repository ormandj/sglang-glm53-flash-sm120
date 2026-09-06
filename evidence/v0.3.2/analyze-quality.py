"""Retain unchanged-grader scores, capped responses and matched question IDs."""
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[1]
spec=importlib.util.spec_from_file_location('grader',repo/'benchmarks/regrade_gsm8k.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
baseline=root.parent/'v0.3.1-release-20260906/qualification'
candidate=root/'qualification'
def gsm(path):
 rows=[json.loads(line) for line in path.read_text().splitlines()]
 mapped={r['conversation_id']:r for r in rows};assert len(mapped)==len(rows)==1319
 return mapped
b=gsm(baseline/'quality/gsm8k-full/accuracy_export.jsonl')
c=gsm(candidate/'quality/gsm8k-full/accuracy_export.jsonl')
assert set(b)==set(c)
correct=lambda r: g._norm(str(r['expected'])) is not None and g.extract(r.get('model_output') or '')==g._norm(str(r['expected']))
assert all(b[k]['expected']==c[k]['expected'] for k in c)
profile=[json.loads(line) for line in (candidate/'quality/gsm8k-full/profile_export.jsonl').read_text().splitlines()]
assert len(profile)==1319
export=json.loads((candidate/'quality/gsm8k-full/profile_export_aiperf.json').read_text())
accounting={
 'requests':len(c),'request_errors':export.get('request_error_rate',{}).get('avg'),
 'error_summary':export['error_summary'],
 'cancelled':sum(bool(r['metadata']['was_cancelled']) for r in profile),
 'pinned_correct':sum(bool(r['passed']) for r in c.values()),
 'glm_correct':sum(correct(r) for r in c.values()),
 'completion_budget':16384,
 'at_completion_budget':[r['metadata']['conversation_id'] for r in profile if r['metrics'].get('usage_completion_tokens',{}).get('value',0)>=16384],
 'pinned_fail_glm_pass':[k for k,r in c.items() if not r['passed'] and correct(r)],
 'pinned_pass_glm_fail':[k for k,r in c.items() if r['passed'] and not correct(r)],
 'grader_sha256':hashlib.sha256((repo/'benchmarks/regrade_gsm8k.py').read_bytes()).hexdigest(),
 'reverse_disagreement_review':'session_000147 states the correct answer 75, but a later bold Last 6 hours label is selected by the GLM-aware extractor. Original score retained.',
 'interpretation':'Both graders are unchanged. All responses, including capped outputs, remain in the denominator. One run per image does not establish quality improvement or equivalence.',
}
paired={'gsm8k':{'baseline_correct':sum(correct(r) for r in b.values()),'candidate_correct':accounting['glm_correct'],'questions':1319,
 'lost':[k for k in c if correct(b[k]) and not correct(c[k])],
 'gained':[k for k in c if not correct(b[k]) and correct(c[k])]}}
for name in ['longctx-73k','longctx-400k']:
 if not (candidate/(name+'.json')).exists():continue
 before=json.loads((baseline/(name+'.json')).read_text());after=json.loads((candidate/(name+'.json')).read_text())
 assert before['prefix_chars']==after['prefix_chars']
 bm={r['idx']:r for r in before['results']};cm={r['idx']:r for r in after['results']}
 assert set(bm)==set(cm) and len(cm)==150
 assert all(bm[k]['expected']==cm[k]['expected'] for k in cm)
 paired[name]={'baseline_correct':before['correct'],'candidate_correct':after['correct'],'questions':150,
 'lost':[k for k in cm if bm[k]['correct'] and not cm[k]['correct']],
 'gained':[k for k in cm if not bm[k]['correct'] and cm[k]['correct']]}
(root/'gsm8k-quality-accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
(root/'paired-quality.json').write_text(json.dumps(paired,indent=2)+'\n')
print(json.dumps({k:v for k,v in accounting.items() if k not in ['pinned_fail_glm_pass']},indent=2))
print(json.dumps(paired,indent=2))
