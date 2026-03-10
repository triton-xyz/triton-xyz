# Mission

- Fix the active `python/tta-ut/` test issues against the target kernel/test corpus under `third_party/triton-ascend/third_party/ascend/unittest/pytest_ut`, using targeted single-test debugging instead of broad unstable reruns.

# Constraints

- Treat `python/tta-ut/setup.sh` as already completed. Do not rerun setup unless the checkout or symlinks are clearly broken.
- Use `python/tta-ut/pytest_one.sh` as the primary debug harness for one failing test at a time, and update its target test selection as needed for the current repro.
- Use `python/tta-ut/pytest.sh` cautiously. The full run can fail abnormally and hang, so it is not the default inner-loop command.
- Default debug outputs are `debug/tmp` for the full suite and `debug/tmp-pytest_one` for single-test runs. Inspect those directories before rerunning broader coverage.
- Focus on repo-owned support and compatibility code such as `python/tta-ut/conftest.py` and `python/tta-ut/torch_npu.py`, plus repo-owned compiler/runtime code needed to make tests pass.
- Skipped test files do not need attention in this mission.
- Validate each claimed fix with a targeted rerun before widening scope.

# Current Strategy

- Start from one concrete failing test already observed in `debug/tmp-0/pytest.log`, repoint `python/tta-ut/pytest_one.sh` to that case, capture fresh logs and IR dumps in `debug/tmp-pytest_one`, then patch the smallest repo-owned compatibility or compiler surface needed to make that case pass before expanding to its neighboring failures.

# Evidence

- 2026-03-11: `python/tta-ut/setup.sh` clones `third_party/triton-ascend` and wires `python/tta-ut/conftest.py` plus `python/tta-ut/torch_npu.py` into `third_party/triton-ascend/third_party/ascend/unittest/pytest_ut/` via symlinks.
- 2026-03-11: `python/tta-ut/pytest.sh` runs `pytest -v third_party/ascend/unittest/pytest_ut` from `third_party/triton-ascend` with `TRITON_ALWAYS_COMPILE=1`, `TRITON_HOME=debug/tmp`, and logs to `debug/tmp/pytest.log`.
- 2026-03-11: `python/tta-ut/pytest_one.sh` is the single-test debug template. It enables `TRITON_DEBUG=1`, `MLIR_ENABLE_DUMP=1`, and `MLIR_ENABLE_DUMP_DIR=debug/tmp-pytest_one/_pass_dump`, and currently points at `test_abs.py` as an example target.
- 2026-03-11: `python/tta-ut/conftest.py` provides pytest-side filtering and compatibility policy, including `SKIP_TEST_FILES`, dtype normalization, and the current supported dtype default of `float32`.
- 2026-03-11: `python/tta-ut/torch_npu.py` supplies the CPU-backed NPU shim by activating `XYZDriver`, remapping selected `torch` allocation APIs away from `npu`, and aliasing `triton.language.extra.cann` and `triton.language.extra.ascend` to `xyz`.
- 2026-03-11: `debug/tmp-0/pytest.log` shows a prior full-suite run collecting 1544 items and then hitting many failures and errors during execution. Representative early failures include `test_broadcast_op.py::test_broadcast_to[float32]`, the `test_cannonicalize_tl_where.py` family, `test_cat_dim.py::test_cat`, and `test_dot.py::*` with `ERROR`.
- 2026-03-11: Because the full-suite run already fails noisily and can hang, the first useful progress comes from isolating one reproducible failing test family rather than rerunning all supported cases.

# Next Options

- Repoint `python/tta-ut/pytest_one.sh` to the first chosen failing case from `debug/tmp-0/pytest.log` and capture a fresh reproducer under `debug/tmp-pytest_one`.
- Group the visible failures into one small family, such as `broadcast`, `cannonicalize_tl_where`, or `dot`, and choose the family with the narrowest likely root cause.
- Inspect `python/tta-ut/conftest.py`, `python/tta-ut/torch_npu.py`, and the relevant repo-owned compiler path for the first isolated failure before considering broader harness changes.
- After fixing one isolated case, rerun only adjacent tests in the same family, not `python/tta-ut/pytest.sh`.

# Blockers

- The broad `python/tta-ut/pytest.sh` path is not a safe tight-loop command because it can hang and mixes many unrelated failures into one noisy log.
- No single failing test has been isolated as the active repro yet, so the next round must choose one concrete case before implementation work starts.
