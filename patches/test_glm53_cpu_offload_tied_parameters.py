"""Build-time contract for GLM-5.3 KDA parameters under CPU offload."""

import inspect

import torch
from torch import nn
from torch.func import functional_call

from sglang.srt.utils.offloader import OffloaderV1


source = inspect.getsource(OffloaderV1.maybe_offload_to_cpu)
assert "tie_weights=False" in source, "CPU offloader does not admit tied KDA aliases"


class TiedParameterModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([2.0]))
        self.attn = nn.Module()
        self.attn.register_parameter("scale", self.scale)

    def forward(self, value):
        return value * self.scale + value * self.attn.scale


module = TiedParameterModule()
device_state = {name: value.clone() for name, value in module.state_dict().items()}
assert set(device_state) == {"scale", "attn.scale"}

try:
    functional_call(module, device_state, (torch.tensor([3.0]),))
except ValueError as error:
    assert "multiple values" in str(error)
else:
    raise AssertionError("negative control did not reject separate values for tied keys")

actual = functional_call(
    module,
    device_state,
    (torch.tensor([3.0]),),
    tie_weights=False,
)
torch.testing.assert_close(actual, torch.tensor([12.0]), rtol=0, atol=0)
print("GLM tied-parameter CPU-offload contract OK")
