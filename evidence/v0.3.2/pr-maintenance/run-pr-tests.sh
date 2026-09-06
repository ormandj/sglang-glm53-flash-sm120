#!/bin/bash
set -euo pipefail
root=/root/.cache/release-validation/pr-maintenance
mkdir -p "$root"
export PYTHONDONTWRITEBYTECODE=1
for n in 37168 37538; do
  cd /tmp/sglang-maintain-$n
  export CUDA_VISIBLE_DEVICES=9 PYTHONPATH=$PWD/python
  if [ "$n" = 37168 ]; then
    files="test/registered/unit/layers/attention/test_dsa_capture_owner.py test/registered/unit/model_executor/runner_backend/test_full_cuda_graph_backend.py"
  else
    files="test/registered/unit/utils/test_extend_mem_profile.py test/registered/unit/utils/test_mem_forensics.py test/registered/unit/model_executor/test_extend_mem_profile_gate.py test/registered/unit/managers/test_mm_embed_mapped_embedder.py"
  fi
  /opt/sglang/bin/python -m pytest -q --tb=short $files > "$root/$n-cpu.log" 2>&1
  echo CPU_PASS_$n
done
while [ ! -e /root/.cache/release-validation/exact-image.status ]; do sleep 5; done
test "$(cat /root/.cache/release-validation/exact-image.status)" = 0
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8
cd /tmp/sglang-maintain-37168
export PYTHONPATH=$PWD/python
/opt/sglang/bin/python -m pytest -q --tb=short test/registered/layers/attention/test_dsa_capture_owner_cuda.py -k 'not deepgemm' > "$root/37168-cuda-owner.log" 2>&1
/opt/sglang/bin/python -m pytest -q --tb=short test/registered/kernels/ops/layernorm/test_mhc_kernels.py > "$root/37168-mhc.log" 2>&1
cd /tmp/sglang-maintain-37625
export PYTHONPATH=$PWD/python
/opt/sglang/bin/python -m pytest -q --tb=short test/registered/kernels/test_dsa_kpool_topk_transform.py > "$root/37625-kpool.log" 2>&1
/opt/sglang/bin/python -m pytest -q --tb=short test/registered/kernels/ops/attention/test_topk_v2.py -k 'oversized or all_equal_overflow or signed_zero' > "$root/37625-jit.log" 2>&1
/opt/sglang/bin/python /tmp/run-pr-topk-aot.py > "$root/37625-aot.log" 2>&1
echo ALL_PR_MAINTENANCE_TESTS_PASSED
