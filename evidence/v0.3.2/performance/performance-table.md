| Workload | Tokens measured | Mean tok/s | Median tok/s | Mean forwards/s | Median forwards/s | Output tok/forward/request, mean / median |
|---|---|---:|---:|---:|---:|---:|
| Decode C1, 5 repetitions | Aggregate output after MTP | 215.5 | 202.6 | 60.77 | 61.38 | 3.60 / 3.42 |
| Decode C2, 5 repetitions | Aggregate output after MTP | 290.2 | 293.8 | 49.11 | 49.16 | 3.02 / 3.08 |
| Decode C3, 5 repetitions | Aggregate output after MTP | 360.3 | 356.6 | 39.20 | 38.98 | 3.11 / 3.12 |
| Decode C4, 5 repetitions | Aggregate output after MTP | 406.9 | 410.2 | 33.31 | 33.62 | 3.06 / 3.03 |
| Cold prefill 8k, C1, 5 requests | Prompt tokens (input) | 5,193.5 | 5,211.0 | n/a | n/a | n/a |
| Cold prefill 32k, C1, 5 requests | Prompt tokens (input) | 5,877.4 | 5,860.4 | n/a | n/a | n/a |
| Cold prefill 64k, C1, 5 requests | Prompt tokens (input) | 5,921.4 | 5,907.2 | n/a | n/a | n/a |
| Cold prefill 128k, C1, 5 requests | Prompt tokens (input) | 5,923.1 | 5,903.7 | n/a | n/a | n/a |

Decode window: average context 17,408-20,480 tokens (16k prompt plus 1k-4k output), 10.7-29.7 seconds per repetition. Decode rates aggregate all C concurrent requests. Prefill rows cover each full cold request to its first token.
