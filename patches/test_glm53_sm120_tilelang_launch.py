"""Build-time contract for the consumer-Blackwell TileLang DSA launch."""

from unittest.mock import patch

import torch

import sglang.kernels.ops.attention.dsa.tilelang_kernel as tilelang_kernel


def capture_launch(sm_major: int) -> dict:
    captured = {}

    def fake_factory(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        def fake_kernel(q, kv, indices, lse):
            return q.new_zeros((q.shape[0], q.shape[1], q.shape[2], 512))

        return fake_kernel

    q = torch.zeros((1, 16, 512), dtype=torch.bfloat16)
    kv = torch.zeros((64, 1, 512), dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 64), dtype=torch.int32)

    with (
        patch.object(tilelang_kernel, "_is_hip", False),
        patch.object(tilelang_kernel, "sparse_attention_fwd_kernel_v1", fake_factory),
        patch.object(torch.cuda, "get_device_capability", return_value=(sm_major, 0)),
    ):
        output = tilelang_kernel.tilelang_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=0.125,
        )

    assert tuple(output.shape) == (1, 1, 16, 512)
    return captured


sm12 = capture_launch(12)
assert sm12["args"] == (16, 512, 0, 64)
assert sm12["kwargs"] == {
    "sm_scale": 0.125,
    "return_lse": False,
    "block_I": 32,
    "num_stages": 1,
    "threads": 128,
}

sm10 = capture_launch(10)
assert sm10["args"] == (16, 512, 0, 64)
assert sm10["kwargs"] == {"sm_scale": 0.125, "return_lse": False}

print("SM120 TileLang DSA launch contract OK")
