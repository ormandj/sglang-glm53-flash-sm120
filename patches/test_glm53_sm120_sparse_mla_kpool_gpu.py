"""SM120 numerical regression for GLM-5.3's KPool-extended DSA table.

Run beside FlashInfer's ``test_sparse_mla_sm120.py`` so this file can reuse
the upstream pack/dequantize/reference helpers without copying them into the
container patch set.  This is an exact-hardware test, not a build-time test.
"""

from __future__ import annotations

import torch

from test_sparse_mla_sm120 import (  # type: ignore[import-not-found]
    _ref_sparse_attn,
    dequantize_kv_dsv4,
    quantize_kv_dsv4,
)


def run_case(num_tokens: int) -> None:
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    torch.manual_seed(53 + num_tokens)
    device = torch.device("cuda")
    num_heads = 16
    d_qk = d_v = 512
    page_size = 64
    num_blocks = 40
    num_kv_tokens = num_blocks * page_size
    base_topk = 2048
    primary_topk = 128
    kpool_tail = 3

    kv_bf16 = (
        torch.randn(
            num_blocks,
            page_size,
            1,
            d_qk,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(
            num_tokens,
            num_heads,
            d_qk,
            dtype=torch.bfloat16,
            device=device,
        )
        / 10.0
    ).clamp(-1, 1)

    # Match GLM-5.3: a fixed 2,048-wide selected table, followed by the
    # index_kpool=4 live tail.  Leave a padding gap before the three live tail
    # slots to prove the adapter must not mask the tail with the base length.
    selected = torch.randint(
        0,
        num_kv_tokens,
        (num_tokens, base_topk),
        dtype=torch.int32,
        device=device,
    )
    selected[:, 1536:] = -1
    tail = torch.randint(
        0,
        num_kv_tokens,
        (num_tokens, kpool_tail),
        dtype=torch.int32,
        device=device,
    )
    indices = torch.cat((selected, tail), dim=-1)
    primary_indices = indices[:, :primary_topk].contiguous()
    extra_indices = indices[:, primary_topk:].contiguous()

    primary_lens = torch.full(
        (num_tokens,),
        primary_indices.shape[-1],
        dtype=torch.int32,
        device=device,
    )
    extra_lens = torch.full(
        (num_tokens,),
        extra_indices.shape[-1],
        dtype=torch.int32,
        device=device,
    )
    sm_scale = d_qk**-0.5

    reference, _ = _ref_sparse_attn(
        q,
        kv_dequant,
        indices,
        sm_scale,
        d_v,
    )
    output = trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=kv_packed,
        workspace_buffer=torch.empty(64 << 20, dtype=torch.uint8, device=device),
        sparse_indices=primary_indices,
        compressed_kv_cache=kv_packed,
        swa_topk_lens=primary_lens,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lens,
        bmm1_scale=sm_scale,
        kv_layout="NHD",
        backend="sparse",
    ).squeeze(1)

    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)
    print(
        f"PASS num_tokens={num_tokens} heads={num_heads} "
        f"primary_topk={primary_topk} extra_topk={extra_indices.shape[-1]}"
    )


if __name__ == "__main__":
    run_case(1)
    run_case(128)
