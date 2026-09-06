"""Render measured decode and prefill distributions without changing analyzers."""
import argparse
import hashlib
import json
from pathlib import Path


def render(gate):
    sources = {}
    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(gate))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    summary = read(gate / 'summary.json')
    rows = []
    for concurrency in range(1, 5):
        cell = summary['decode'][f'c{concurrency}']
        assert len(cell['repetitions']) == 5
        windows = []
        for repetition in cell['repetitions']:
            analysis = read(gate / 'decode' / f'c{concurrency}' / repetition['id'] / 'decode-analysis.json')
            assert analysis['validation']['valid'] is True
            windows.append({'repetition': repetition['id'], **analysis['plateau']})
        rows.append({
            'workload': f'Decode C{concurrency}, 5 repetitions',
            'tokens_measured': 'Aggregate output after MTP',
            'samples': 5,
            'sample_kind': 'Concurrent cohort repetitions',
            'concurrency': concurrency,
            'definition': 'Aggregate emitted output across the active cohort, including reasoning and content, over the selected steady decode window.',
            'windows': windows,
            'tokens_per_second': cell['synthetic_decode_tokens_per_second'],
            'forward_passes_per_second': cell['engine_forward_passes_per_second'],
            'output_tokens_per_forward_per_request': cell['output_tokens_per_forward_per_request'],
        })
    latency = []
    for length in ['8k', '32k', '64k', '128k']:
        cell = gate / 'prefill' / f'{length}-c1'
        analysis = read(cell / 'prefill-analysis.json')
        assert analysis['validation']['valid'] is True
        profile = read(cell / 'profile_export_aiperf.json')
        rate = profile['prefill_throughput_per_user']
        assert rate['count'] == 5
        rows.append({
            'workload': f'Cold prefill {length}, C1, 5 requests',
            'tokens_measured': 'Prompt tokens (input)',
            'samples': rate['count'],
            'sample_kind': 'Individual cold requests',
            'concurrency': 1,
            'tokens_per_second': {'mean': rate['avg'], 'median': rate['p50']},
            'definition': 'Individual request prompt tokens divided by time to first token; mean and median over five requests.',
        })
        ttft = analysis['requests']['time_to_first_token_ms']
        latency.append({'length': length, 'mean_ttft_seconds': ttft['mean']/1000,
                        'median_ttft_seconds': ttft['median']/1000,
                        'aggregate_prompt_tokens_per_second': analysis['requests']['aggregate_prompt_tokens_per_second']})
    table = ['| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        rate = row['tokens_per_second']
        forward = row.get('forward_passes_per_second')
        accepted = row.get('output_tokens_per_forward_per_request')
        fields = [row['workload'], row['tokens_measured'], f"{rate['mean']:,.1f}", f"{rate['median']:,.1f}"]
        fields += [f"{forward['mean']:.2f}", f"{forward['median']:.2f}", f"{accepted['mean']:.2f} / {accepted['median']:.2f}"] if forward else ['n/a', 'n/a', 'n/a']
        table.append('| ' + ' | '.join(fields) + ' |')
    windows = [window for row in rows if 'windows' in row for window in row['windows']]
    bands = {(window['average_context_lower'], window['average_context_upper']) for window in windows}
    assert len(bands) == 1
    lower, upper = next(iter(bands))
    durations = [window['duration_seconds'] for window in windows]
    table += ['', f'Decode window: average context {lower:,.0f}-{upper:,.0f} tokens (16k prompt plus 1k-4k output), {min(durations):.1f}-{max(durations):.1f} seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.']
    latency_table = ['| Cold prefill, C1 | Mean TTFT | Median TTFT | Aggregate prompt tok/s over the cell |', '|---|---:|---:|---:|']
    for row in latency:
        latency_table.append(f"| {row['length']} | {row['mean_ttft_seconds']:.3f} s | {row['median_ttft_seconds']:.3f} s | {row['aggregate_prompt_tokens_per_second']:,.1f} |")
    return {'rows': rows, 'prefill_latency': latency, 'source_sha256': sources}, '\n'.join(table)+'\n', '\n'.join(latency_table)+'\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data, throughput, latency = render(args.gate)
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / 'performance-table.json').write_text(json.dumps(data, indent=2)+'\n')
    (args.output / 'performance-table.md').write_text(throughput)
    (args.output / 'prefill-latency-table.md').write_text(latency)
    print(throughput+'\n'+latency)
