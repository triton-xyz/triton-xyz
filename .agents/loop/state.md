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

- Keep `python/tta-ut/pytest_one.sh` pointed at the current isolated repro while validating small repo-owned shim fixes first, then inspect repo-owned Triton frontend or compiler behavior when the failure comes from compilation rather than pytest-side compatibility.

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

# Next Options

- Inspect whether the power-of-two restriction in `third_party/triton/python/triton/_utils.py::validate_block_shape` or the local frontend path around `tl.make_block_ptr` is stricter than the Ascend corpus expects for `test_advance.py`.
- Check whether the `advance` family should be handled by a repo-owned compatibility shim, a frontend relaxation for the XYZ path, or by routing to a different lowering path before changing broader compiler semantics.
- Watch for more `pytest_ut` files that omit `import torch_npu`; if they now pass automatically, keep the eager shim import in `python/tta-ut/conftest.py` and avoid expanding it further without concrete failures.
- Continue avoiding `python/tta-ut/pytest.sh` until several neighboring isolated failures have been cleared.

# Blockers

- The broad `python/tta-ut/pytest.sh` path is not a safe tight-loop command because it can hang and mixes many unrelated failures into one noisy log.
- The remaining failure set is still large and heterogeneous, so the next round must keep choosing one narrow repro at a time rather than inferring a shared root cause from the noisy full-suite log.
- The newly isolated `test_advance` blocker currently points into vendored Triton frontend validation (`third_party/triton/python/triton/_utils.py`), so the next round needs to decide whether there is a repo-owned interception point before considering any vendored change.
