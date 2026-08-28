"""Source and GPU regression for GLM-5.3's no-RoPE MLA reserved slot."""

import inspect

import torch

import sglang.kernels.ops.kvcache.mla_buffer as mla_buffer


module_source = inspect.getsource(mla_buffer)
kernel_start = module_source.index("def set_mla_kv_buffer_kernel_norope(")
kernel_end = module_source.index("\n\n# Above this loc count", kernel_start)
kernel_source = module_source[kernel_start:kernel_end]
assert "reserved_skip_index" in kernel_source
assert "DCP_RANK: tl.constexpr" in kernel_source
assert "DCP_WORLD_SIZE: tl.constexpr" in kernel_source
assert "loc != reserved_skip_index" in kernel_source
assert "loc % DCP_WORLD_SIZE == DCP_RANK" in kernel_source
assert "mask=mask & is_valid" in kernel_source

dispatcher_source = inspect.getsource(mla_buffer.set_mla_kv_buffer_triton)
assert "set_mla_kv_buffer_kernel_norope[grid]" in dispatcher_source
assert "reserved_skip_index," in dispatcher_source
assert "DCP_RANK=get_parallel().attn_dcp_rank" in dispatcher_source
assert "DCP_WORLD_SIZE=get_parallel().attn_dcp_size" in dispatcher_source

if not torch.cuda.is_available():
    print("GLM-5.3 no-RoPE reserved-slot source contract OK; GPU test skipped")
    raise SystemExit(0)

device = torch.device("cuda")
kv_buffer = torch.full((4, 1, 528), 0x5A, dtype=torch.uint8, device=device)
slot0_before = kv_buffer[0].clone()
cache_k_nope = torch.empty((2, 1, 528), dtype=torch.uint8, device=device)
cache_k_nope[0].fill_(0xA5)
cache_k_nope[1].fill_(0x3C)
loc = torch.tensor([0, 2], dtype=torch.int64, device=device)

mla_buffer.set_mla_kv_buffer_triton(
    kv_buffer,
    loc,
    cache_k_nope,
    None,
    reserved_skip_index=0,
)
torch.cuda.synchronize()
assert torch.equal(kv_buffer[0], slot0_before)
assert torch.equal(kv_buffer[2], cache_k_nope[1])

mla_buffer.set_mla_kv_buffer_triton(
    kv_buffer,
    loc[:1],
    cache_k_nope[:1],
    None,
    reserved_skip_index=-1,
)
torch.cuda.synchronize()
assert torch.equal(kv_buffer[0], cache_k_nope[0])

print("GLM-5.3 no-RoPE reserved-slot GPU contract OK")
