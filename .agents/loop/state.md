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

- Keep `python/tta-ut/pytest_one.sh` pointed at the isolated `test_advance.py` repro, use repo-owned pytest shims to mirror Ascend frontend expectations when possible, and stop short of vendored edits until the remaining failure proves there is no repo-owned interception point before the upstream Triton verifier.

# Evidence

- 2026-03-11: `python/tta-ut/setup.sh` clones `third_party/triton-ascend` and wires `python/tta-ut/conftest.py` plus `python/tta-ut/torch_npu.py` into `third_party/triton-ascend/third_party/ascend/unittest/pytest_ut/` via symlinks.
- 2026-03-11: `python/tta-ut/pytest.sh` runs `pytest -v third_party/ascend/unittest/pytest_ut` from `third_party/triton-ascend` with `TRITON_ALWAYS_COMPILE=1`, `TRITON_HOME=debug/tmp`, and logs to `debug/tmp/pytest.log`.
- 2026-03-11: `python/tta-ut/pytest_one.sh` is the single-test debug template. It enables `TRITON_DEBUG=1`, `MLIR_ENABLE_DUMP=1`, and `MLIR_ENABLE_DUMP_DIR=debug/tmp-pytest_one/_pass_dump`. For the current repro it now targets `test_address_check.py -k test_cpu_tensor_should_fail`.
- 2026-03-11: `python/tta-ut/conftest.py` provides pytest-side filtering and compatibility policy, including `SKIP_TEST_FILES`, dtype normalization, and the current supported dtype default of `float32`.
- 2026-03-11: `python/tta-ut/torch_npu.py` supplies the CPU-backed NPU shim by activating `XYZDriver`, remapping selected `torch` allocation APIs away from `npu`, and aliasing `triton.language.extra.cann` and `triton.language.extra.ascend` to `xyz`.
- 2026-03-11: `debug/tmp-0/pytest.log` shows a prior full-suite run collecting 1544 items and then hitting many failures and errors during execution. Representative early failures include `test_broadcast_op.py::test_broadcast_to[float32]`, the `test_cannonicalize_tl_where.py` family, `test_cat_dim.py::test_cat`, and `test_dot.py::*` with `ERROR`.
- 2026-03-11: Because the full-suite run already fails noisily and can hang, the first useful progress comes from isolating one reproducible failing test family rather than rerunning all supported cases.
- 2026-03-11: A targeted rerun of `python/tta-ut/pytest_one.sh` reproduced `test_address_check.py::test_cpu_tensor_should_fail` as `Failed: DID NOT RAISE <class 'ValueError'>`, confirming the fake NPU shim was allowing plain CPU tensors through kernel launch.
- 2026-03-11: `python/tta-ut/torch_npu.py` now tags tensors created through fake-NPU entrypoints (`device='npu'`, `.npu()`, and common factory helpers that inherit fake-NPU placement) and patches the local `xyz` launcher memref builder to raise `ValueError("Pointer argument cannot be accessed from Triton (cpu tensor?)")` for untagged CPU tensors.
- 2026-03-11: Validation succeeded with `bash python/tta-ut/pytest_one.sh` for `test_address_check.py::test_cpu_tensor_should_fail`, `pytest -v third_party/ascend/unittest/pytest_ut/test_address_check.py` for the whole file (`2 passed`), and a sanity rerun of the previously passing `test_abs.py::test_case[param_list1]` (`1 passed`).
- 2026-03-11: A targeted probe of `test_annotations.py::test_int_annotation[False-8]` first failed before compilation with `RuntimeError: ... device string: npu`, showing some tests rely on `conftest.py` to activate the fake-NPU shim even when they do not import `torch_npu` themselves.
- 2026-03-11: `python/tta-ut/conftest.py` now imports `torch_npu` eagerly and patches Triton's `AsmDict.__getitem__` for `ttir` so function signature arguments are normalized back to `%argN` form in test-visible TTIR strings, matching the Ascend test corpus expectation without changing compiler IR generation globally.
- 2026-03-11: Validation succeeded with `bash python/tta-ut/pytest_one.sh` for `test_annotations.py::test_int_annotation[False-8]` and `pytest -v third_party/ascend/unittest/pytest_ut/test_annotations.py` for the whole file (`10 passed`).
- 2026-03-11: `python/tta-ut/pytest_one.sh` is now repointed to `test_advance.py -k test_advance_with_boundary_check[shape0-float32]`, and the harness reproduces a compile-time blocker: `tl.make_block_ptr` rejects block shape `(33, 9, 2)` with `ValueError: Shape element 0 must be a power of 2` from `third_party/triton/python/triton/_utils.py:68`.
- 2026-03-11: Comparing `third_party/triton/python/triton/_utils.py` against `third_party/triton-ascend/python/triton/language/_utils.py` showed the Ascend fork intentionally comments out the block-shape power-of-two check, so the active repro was failing in stricter upstream Triton frontend validation rather than in repo-owned TTA lowering.
- 2026-03-11: `python/tta-ut/torch_npu.py` now patches both `triton._utils.validate_block_shape` and `triton.language.core.validate_block_shape` to match the Ascend fork during fake-NPU pytest runs, which moves `test_advance.py` past the earlier `tl.make_block_ptr` frontend exception without editing vendored Triton sources.
- 2026-03-11: After the shim patch, `bash python/tta-ut/pytest_one.sh` for `test_advance_with_boundary_check[shape0-float32]` still fails, but now at compiler verification with `Number of elements must be power-of-two` on `tt.load` from `tensor<33x9x2xf32>`, proving the next blocker is deeper than pytest compatibility.
- 2026-03-11: `cd third_party/triton-ascend && TTX_PYTEST_DTYPE=float32 pytest -v third_party/ascend/unittest/pytest_ut/test_advance.py` selected 8 runnable cases and finished `3 passed, 5 failed, 16 skipped`; the remaining failures are the non-power-of-two `tt.load` verifier cases for shapes `(33, 9, 6)`, `(1, 3)`, `(3, 1)`, `(1, 13)`, and `(13, 1)`.

# Next Options

- Inspect whether the `test_advance.py` harness can be routed to the Ascend-flavored Triton frontend/runtime selection without vendored edits, since the current active path still loads the stricter upstream verifier from `third_party/triton`.
- Compare the active verifier failure against `third_party/triton-ascend/lib/Dialect/Triton/IR/Traits.cpp`, where the relevant power-of-two checks are commented out, and determine whether there is a repo-owned build or package-selection hook that can reuse that behavior.
- If there is no repo-owned interception point, decide whether this mission now justifies a scoped vendored change in the active Triton verifier path for block-pointer `tt.load`/`tt.store` with non-power-of-two tensor shapes.
- Keep watching for more `pytest_ut` files that rely on the relaxed block-shape frontend behavior, because the new shim patch may unblock other cases even before the deeper verifier issue is solved.

# Blockers

- The broad `python/tta-ut/pytest.sh` path is not a safe tight-loop command because it can hang and mixes many unrelated failures into one noisy log.
- The remaining failure set is still large and heterogeneous, so the next round must keep choosing one narrow repro at a time rather than inferring a shared root cause from the noisy full-suite log.
- The active `test_advance` blocker has moved past pytest/frontend shims and now fails inside the upstream Triton verifier path loaded from `third_party/triton`, where non-power-of-two tensor element counts on `tt.load` are rejected before the repo's local lowering pipeline can help.
