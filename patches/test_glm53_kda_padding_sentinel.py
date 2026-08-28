"""Build-time contract for padded KDA rows and the Mamba state pool."""

import inspect

from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel


source = inspect.getsource(TritonKDAKernel.extend)
assert "initial_state_indices=cache_indices" in source
assert "ssm_states.shape[0] - 1" not in source
assert "ssm_cache_indices" not in source

# MambaPool has size + 1 rows: slot zero is reserved, while the final row is a
# live allocatable slot. A padded -1 row must therefore remain -1 instead of
# aliasing the last live request state.
pool_size = 5
pool_rows = pool_size + 1
last_live_slot = pool_size
old_remap = pool_rows - 1
assert old_remap == last_live_slot
padding_slot = -1
assert padding_slot != last_live_slot
print("GLM KDA padding-sentinel contract OK")
