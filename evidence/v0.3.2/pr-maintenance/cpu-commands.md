# Reproducing the additional CPU checks

Run each command in its corresponding PR source checkout with the qualified image dependencies. These commands document the test files and GPU-hiding setting; the retained log files record the completed staged-snapshot results.

```bash
# PR #35954
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/scheduler/test_min_free_slots_delayer.py test/registered/unit/server_args/test_server_args.py
# PR #32815
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/unit/test_model_overrides.py
# PR #32686
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/unit/layers/deep_gemm_wrapper/test_compile_utils.py
```
