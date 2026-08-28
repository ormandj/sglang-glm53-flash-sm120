"""Build-time contract for derived MLA tensors under CPU offload."""

import inspect

import torch
from torch import nn
from torch.func import functional_call

from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.utils.offloader import OffloaderV1


offloader_source = inspect.getsource(OffloaderV1.maybe_offload_to_cpu)
assert "module.named_buffers(remove_duplicate=False)" in offloader_source
assert "if k not in device_state" in offloader_source

mla_source = inspect.getsource(DeepseekV2AttentionMLA.__init__)
assert 'register_buffer("w_kc", None, persistent=False)' in mla_source
assert 'register_buffer("w_vc", None, persistent=False)' in mla_source


class DerivedExecutionBufferModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("derived", None, persistent=False)

    def forward(self, value):
        return value * self.weight + value * self.derived


module = DerivedExecutionBufferModule()
module.derived = torch.tensor([3.0])

# Derived execution buffers must not become checkpoint state.
assert set(module.state_dict()) == {"weight"}
buffers = dict(module.named_buffers(remove_duplicate=False))
assert set(buffers) == {"derived"}
assert buffers["derived"] is module.derived

# Mirror OffloaderV1's device-state assembly and prove functional_call accepts
# the non-persistent buffer while restoring the module's original value.
device_state = {name: value.clone() for name, value in module.state_dict().items()}
for name, value in module.named_buffers(remove_duplicate=False):
    if name not in device_state:
        device_state[name] = value.clone()
device_state["derived"] = torch.tensor([5.0])

actual = functional_call(
    module,
    device_state,
    (torch.tensor([4.0]),),
    tie_weights=False,
)
torch.testing.assert_close(actual, torch.tensor([28.0]), rtol=0, atol=0)
torch.testing.assert_close(module.derived, torch.tensor([3.0]), rtol=0, atol=0)

print("GLM derived-MLA-buffer CPU-offload contract OK")
