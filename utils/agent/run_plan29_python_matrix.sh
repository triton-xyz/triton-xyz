#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_RUNNER="$SCRIPT_DIR/run_kernel_case_template.sh"
PYTEST_RUNNER="$SCRIPT_DIR/run_pytest_case_template.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

run_kernel_case() {
  local dump_dir="$1"
  local kernel_py="$2"
  shift 2
  AGENT_DUMP_DIR="$dump_dir" KERNEL_PY="$kernel_py" bash "$KERNEL_RUNNER" "$@"
}

run_pytest_case() {
  local dump_dir="$1"
  local pytest_target="$2"
  shift 2
  AGENT_DUMP_DIR="$dump_dir" PYTEST_TARGET="$pytest_target" bash "$PYTEST_RUNNER" -q "$@"
}

run_kernel_case "plan29_vec_add" "python/tests/test_vec_add.py"

run_pytest_case "plan29_structured_basic_addptr_1d" \
  "python/tests/test_triton_to_structured.py::test_basic_addptr_1d"
run_pytest_case "plan29_structured_masked_1d_neg3" \
  "python/tests/test_triton_to_structured.py::test_masked_1d[-3]"
run_pytest_case "plan29_structured_masked_1d_0" \
  "python/tests/test_triton_to_structured.py::test_masked_1d[0]"
run_pytest_case "plan29_structured_masked_1d_3" \
  "python/tests/test_triton_to_structured.py::test_masked_1d[3]"
run_pytest_case "plan29_structured_masked_1d_8" \
  "python/tests/test_triton_to_structured.py::test_masked_1d[8]"
run_pytest_case "plan29_structured_masked_1d_13" \
  "python/tests/test_triton_to_structured.py::test_masked_1d[13]"
run_pytest_case "plan29_structured_block_ptr_basic" \
  "python/tests/test_triton_to_structured.py::test_block_ptr_basic"
run_pytest_case "plan29_structured_gather_scatter_2d" \
  "python/tests/test_triton_to_structured.py::test_gather_scatter_2d"
run_pytest_case "plan29_structured_scalar_addptr_splat_0" \
  "python/tests/test_triton_to_structured.py::test_scalar_addptr_splat[0]"
run_pytest_case "plan29_structured_scalar_addptr_splat_2" \
  "python/tests/test_triton_to_structured.py::test_scalar_addptr_splat[2]"
run_pytest_case "plan29_structured_scalar_addptr_splat_5" \
  "python/tests/test_triton_to_structured.py::test_scalar_addptr_splat[5]"
run_pytest_case "plan29_structured_row_major_2d" \
  "python/tests/test_triton_to_structured.py::test_row_major_2d"

run_pytest_case "plan29_unstructured_masked_gather_scatter" \
  "python/tests/test_triton_to_unstructured.py::test_masked_gather_scatter"
run_pytest_case "plan29_unstructured_offset_width_upgrade" \
  "python/tests/test_triton_to_unstructured.py::test_offset_width_upgrade"
run_pytest_case "plan29_unstructured_loop_ptr_iter_args" \
  "python/tests/test_triton_to_unstructured.py::test_loop_ptr_iter_args"
run_pytest_case "plan29_unstructured_make_tensor_ptr_add_base" \
  "python/tests/test_triton_to_unstructured.py::test_make_tensor_ptr_add_base"

run_pytest_case "plan29_ptr_regular_copy" \
  "python/tests/test_ptr.py::test_regular_copy"
run_pytest_case "plan29_ptr_pointer_select" \
  "python/tests/test_ptr.py::test_pointer_select"

if [[ -f "$REPO_ROOT/build/libproton.so" ]]; then
  run_pytest_case "plan29_cpu_proton_threads" \
    "python/tests/test_cpu_proton_threads.py::test_cpu_trace_uses_multiple_thread_lanes"
else
  echo "[plan29-matrix] skip python/tests/test_cpu_proton_threads.py: build/libproton.so not found"
fi
