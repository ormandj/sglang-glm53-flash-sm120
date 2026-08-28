"""Build-time and SM120 GPU contract for GLM-5.3's unfused KPool transform."""

import inspect

import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_fast,
    transform_index_page_table_decode_ref,
    transform_index_page_table_prefill_fast,
    transform_index_page_table_prefill_ref,
)


decode_source = inspect.getsource(transform_index_page_table_decode_fast)
prefill_source = inspect.getsource(transform_index_page_table_prefill_fast)
assert "shape[1] in (2048, 2051)" in decode_source
assert "shape[1] in (2048, 2051)" in prefill_source

if not torch.cuda.is_available():
    print("GLM-5.3 2,051-entry unfused KPool source contract OK; GPU test skipped")
    raise SystemExit(0)

device = torch.device("cuda")
context_length = 4096
base = torch.arange(context_length, dtype=torch.int32, device=device)
decode_page_table = torch.stack((base + 10000, base + 20000))

indices = torch.arange(2051, dtype=torch.int32, device=device).repeat(2, 1)
indices[:, 1536:2048] = -1
indices[0, 2048:2051] = torch.tensor([3000, 3500, 4095], device=device)
indices[1, 2048:2051] = torch.tensor([4095, 3500, 3000], device=device)

decode_expected = transform_index_page_table_decode_ref(decode_page_table, indices)
decode_actual = transform_index_page_table_decode_fast(decode_page_table, indices)
torch.testing.assert_close(decode_actual, decode_expected, rtol=0, atol=0)
assert decode_actual[0, 2048:2051].tolist() == [13000, 13500, 14095]
assert decode_actual[1, 2048:2051].tolist() == [24095, 23500, 23000]

prefill_page_table = base.add(30000).unsqueeze(0)
prefill_expected = transform_index_page_table_prefill_ref(
    prefill_page_table,
    indices,
    extend_lens_cpu=[2],
)
prefill_actual = transform_index_page_table_prefill_fast(
    prefill_page_table,
    indices,
    extend_lens_cpu=[2],
)
torch.testing.assert_close(prefill_actual, prefill_expected, rtol=0, atol=0)
assert prefill_actual[0, 2048:2051].tolist() == [33000, 33500, 34095]
assert prefill_actual[1, 2048:2051].tolist() == [34095, 33500, 33000]

print("GLM-5.3 2,051-entry unfused KPool decode and prefill GPU contracts OK")
