"""SM120 numerical regression for GLM-5.3's no-RoPE KPool DSA table.

Run beside FlashInfer's test_sparse_mla_sm120.py so this file can reuse the
patched pack/dequantize/reference helpers. This is an exact-hardware test,
not a build-time test.
"""

from __future__ import annotations

import torch

from test_sparse_mla_sm120 import (  # type: ignore[import-not-found]
    _ref_sparse_attn,
    dequantize_kv_glm_next_nope,
    quantize_kv_glm_next_nope,
)


def run_case(num_tokens: int) -> None:
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    torch.manual_seed(53 + num_tokens)
    device = torch.device("cuda")
    num_heads = 32
    d_qk = d_v = 512
    page_size = 64
    num_blocks = 40
    num_kv_tokens = num_blocks * page_size
    base_topk = 2048
    kpool_tail = 3
    kernel_topk = 2176

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
    kv_packed = quantize_kv_glm_next_nope(kv_bf16)
    kv_dequant = dequantize_kv_glm_next_nope(kv_packed)
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

    # Match GLM-5.3: a fixed 2,048-wide selected table followed by the
    # index_kpool=4 live tail. Leave a padding gap before the three live tail
    # slots to prove that a base sequence length must not mask the tail.
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
    padded = torch.nn.functional.pad(
        indices, (0, kernel_topk - indices.shape[-1]), value=-1
    ).contiguous()
    full_lens = torch.full(
        (num_tokens,), kernel_topk, dtype=torch.int32, device=device
    )
    sm_scale = d_qk**-0.5

    reference, _ = _ref_sparse_attn(q, kv_dequant, padded, sm_scale, d_v)
    output = trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv_packed.transpose(1, 2),
        workspace_buffer=torch.empty(64 << 20, dtype=torch.uint8, device=device),
        qk_nope_head_dim=256,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        block_tables=padded.unsqueeze(1),
        seq_lens=full_lens,
        sparse_mla_top_k_lens=full_lens,
        max_seq_len=kernel_topk,
        sparse_mla_top_k=kernel_topk,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        kv_scale_format="arbitrary_fp32",
        backend="sparse",
    ).squeeze(1)

    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)
    print(
        f"PASS num_tokens={num_tokens} heads={num_heads} "
        f"input_topk={indices.shape[-1]} kernel_topk={kernel_topk}"
    )


if __name__ == "__main__":
    run_case(1)
    run_case(128)
