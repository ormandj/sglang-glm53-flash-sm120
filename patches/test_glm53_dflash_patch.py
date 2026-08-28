from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


model = Glm5NextModel.__new__(Glm5NextModel)
nn.Module.__init__(model)
model.config = SimpleNamespace(mhc=True, hc_mult=4)
model.dflash_capture = True
hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
residual = torch.full_like(hidden_states, 2)
actual = model._prepare_aux_hidden_state(hidden_states, residual)
expected = (hidden_states + residual).unflatten(-1, (4, -1)).mean(dim=-2)
torch.testing.assert_close(actual, expected)
assert actual.shape == (2, 3)

model.dflash_capture = False
torch.testing.assert_close(
    model._prepare_aux_hidden_state(hidden_states, residual), hidden_states + residual
)

wrapper = Glm5NextForConditionalGeneration.__new__(
    Glm5NextForConditionalGeneration
)
nn.Module.__init__(wrapper)
wrapper.pp_group = SimpleNamespace(is_last_rank=True)
wrapper.model = SimpleNamespace(dflash_capture=False, layers_to_capture=[])
wrapper.capture_aux_hidden_states = False
wrapper.set_dflash_layers_to_capture([5, 14, 24, 33, 42])
assert wrapper.capture_aux_hidden_states
assert wrapper.model.dflash_capture
assert wrapper.model.layers_to_capture == [6, 15, 25, 34, 43]

print("GLM-5.3 DFlash2 mHC hidden-state capture contract valid")
