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

- Keep `python/tta-ut/pytest_one.sh` on one isolated repro at a time, and prefer repo-owned Python compatibility shims first when a failure is caused by frontend behavior differences between active `third_party/triton` and the Ascend test corpus. Keep `python/tta-ut/patch_active_triton_verifier.sh` available for the separate verifier-path mismatch already proven by `test_advance.py`.

# Evidence

- 2026-03-11: `python/tta-ut/setup.sh` clones `third_party/triton-ascend` and wires `python/tta-ut/conftest.py` plus `python/tta-ut/torch_npu.py` into `third_party/triton-ascend/third_party/ascend/unittest/pytest_ut/` via symlinks.
- 2026-03-11: `python/tta-ut/pytest.sh` runs `pytest -v third_party/ascend/unittest/pytest_ut` from `third_party/triton-ascend` with `TRITON_ALWAYS_COMPILE=1`, `TRITON_HOME=debug/tmp`, and logs to `debug/tmp/pytest.log`.
- 2026-03-11: `python/tta-ut/pytest_one.sh` is the single-test debug template. It enables `TRITON_DEBUG=1`, `MLIR_ENABLE_DUMP=1`, and `MLIR_ENABLE_DUMP_DIR=debug/tmp-pytest_one/_pass_dump`. For the current repro it now targets `test_advance.py -k test_advance_with_boundary_check[shape0-float32]`.
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
- 2026-03-11: A repo-owned Python shim that skipped the initial `module.verify()` check was not sufficient; the same non-power-of-two invariant still fired later inside the MLIR pass manager and then again inside `build/bin/triton-xyz-opt`, so the remaining blocker was in the active C++ verifier path, not just Python frontend glue.
- 2026-03-11: `third_party/triton/python/triton/_C/libtriton.so` resolves to `build/libtriton.so`, so changing `third_party/triton/lib/Dialect/Triton/IR/Traits.cpp` and rebuilding `libtriton.so` plus `triton-xyz-opt` updates the exact binaries used by the fake-NPU pytest flow.
- 2026-03-11: Relaxing the power-of-two element-count checks in the active `third_party/triton/lib/Dialect/Triton/IR/Traits.cpp` to match `third_party/triton-ascend/lib/Dialect/Triton/IR/Traits.cpp`, then rebuilding `libtriton.so` and `triton-xyz-opt`, fixed the isolated `test_advance_with_boundary_check[shape0-float32]` repro (`1 passed`).
- 2026-03-11: `cd third_party/triton-ascend && TTX_PYTEST_DTYPE=float32 pytest -v third_party/ascend/unittest/pytest_ut/test_advance.py` now finishes `8 passed, 16 skipped`, clearing the previously failing float32 `test_advance.py` family.
- 2026-03-11: `python/tta-ut/patch_active_triton_verifier.sh` now codifies the active Triton verifier patch-and-rebuild flow for future rounds, because the `third_party/triton/` checkout is currently outside git tracking in this repository.
- 2026-03-11: Repointing `python/tta-ut/pytest_one.sh` to `test_broadcast_op.py -k test_broadcast_to[float32]` reproduced a different frontend blocker: active `third_party/triton/python/triton/compiler/code_generator.py` rejected module-level globals like `NUMEL : tl.constexpr = XS * ZS` with `NameError("Cannot access global variable NUMEL... annotating a variable as constexpr ... is not supported")`.
- 2026-03-11: Comparing active Triton against `third_party/triton-ascend/python/triton/compiler/code_generator.py` showed the Ascend fork treats `__annotations__[name] == tl.constexpr` as constexpr-global metadata, while the active frontend only accepts `triton.language.constexpr(...)` values.
- 2026-03-11: `python/tta-ut/torch_npu.py` now patches `triton.compiler.code_generator.CodeGenerator._is_constexpr_global` to honor module-level `tl.constexpr` annotations during fake-NPU pytest runs, matching the Ascend corpus expectation without editing the active vendored Triton Python sources.
- 2026-03-11: Validation succeeded with `bash python/tta-ut/pytest_one.sh` for `test_broadcast_op.py::test_broadcast_to[float32]`, `pytest -v third_party/ascend/unittest/pytest_ut/test_broadcast_op.py` for the whole file (`1 passed`), and a second annotated-constexpr sanity probe `pytest -v third_party/ascend/unittest/pytest_ut/test_template.py` (`1 passed`).

# Next Options

- Retarget `python/tta-ut/pytest_one.sh` to the next earliest reproducible non-skipped failure from the saved full-suite evidence, likely one of the early elementwise files such as `test_acos.py` or `test_acosh.py`, and keep the inner loop narrow.
- Watch for other `pytest_ut` files that now pass automatically under the new annotated-constexpr shim, especially kernels that use module-level `X_SIZE : tl.constexpr = ...` style globals inside `@triton.jit` bodies.
- Watch for files that now pass automatically under the relaxed active Triton verifier, especially tests that rely on non-power-of-two block-pointer shapes or tensor pointer rewrites.
- When a new failure looks compiler-specific rather than pytest-shim-specific, confirm first whether it reproduces after rerunning `python/tta-ut/patch_active_triton_verifier.sh` to eliminate stale local binaries.

# Blockers

- The broad `python/tta-ut/pytest.sh` path is not a safe tight-loop command because it can hang and mixes many unrelated failures into one noisy log.
- The remaining failure set is still large and heterogeneous, so the next round must keep choosing one narrow repro at a time rather than inferring a shared root cause from the noisy full-suite log.
- The active `third_party/triton/` source tree used by the pytest flow is currently untracked in git, so durable preservation inside this repository needs tracked helper scripts and state updates rather than relying on the local checkout diff alone.
