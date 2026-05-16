#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTEST_RUNNER="$SCRIPT_DIR/run_pytest_case_template.sh"

run_pytest_case() {
  local dump_dir="$1"
  local pytest_target="$2"
  shift 2
  AGENT_DUMP_DIR="$dump_dir" PYTEST_TARGET="$pytest_target" bash "$PYTEST_RUNNER" -q "$@"
}

run_pytest_case "plan30_tta_to_memref_atomic_add_xchg_true" \
  "python/tests/test_tta_to_memref.py::test_atomic_add_xchg_masked_true"
run_pytest_case "plan30_tta_to_memref_atomic_add_xchg_false" \
  "python/tests/test_tta_to_memref.py::test_atomic_add_xchg_masked_false"
run_pytest_case "plan30_tta_to_memref_atomic_cas_scalar" \
  "python/tests/test_tta_to_memref.py::test_atomic_cas_scalar"
run_pytest_case "plan30_tta_to_memref_indirect_reindex_2d" \
  "python/tests/test_tta_to_memref.py::test_indirect_reindex_2d"
run_pytest_case "plan30_tta_to_memref_from_tt_ptr_indirect" \
  "python/tests/test_tta_to_memref.py::test_from_tt_ptr_indirect"
run_pytest_case "plan30_tta_to_memref_loop_indirect_seed" \
  "python/tests/test_tta_to_memref.py::test_loop_indirect_seed"
run_pytest_case "plan30_tta_to_memref_loop_indirect_recurrence" \
  "python/tests/test_tta_to_memref.py::test_loop_indirect_recurrence"
run_pytest_case "plan30_tta_to_memref_loop_derived_indirect_use" \
  "python/tests/test_tta_to_memref.py::test_loop_derived_indirect_use"
run_pytest_case "plan30_tta_to_memref_loop_indirect_dynamic_lower_bound" \
  "python/tests/test_tta_to_memref.py::test_loop_indirect_no_seed_dynamic_lower_bound"
run_pytest_case "plan30_tta_to_memref_loop_indirect_dynamic_step" \
  "python/tests/test_tta_to_memref.py::test_loop_indirect_no_seed_dynamic_step"
run_pytest_case "plan30_tta_to_memref_loop_indirect_non_zero_direct_step" \
  "python/tests/test_tta_to_memref.py::test_loop_indirect_no_seed_non_zero_direct_step"
run_pytest_case "plan30_tta_to_memref_wrap_dynamic_mask" \
  "python/tests/test_tta_to_memref.py::test_wrap_dynamic_mask"
