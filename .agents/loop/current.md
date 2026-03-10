# Current

Keep this file short. It is the live working view for the next few rounds.

## Goal

- Fix issues in tests under `python/tta-ut/`.
- The target kernels under test are in `third_party/triton-ascend/third_party/ascend/unittest/pytest_ut`.

## Constraints

- Do not rerun `python/tta-ut/setup.sh`; the environment is already configured.
- Be careful with `bash python/tta-ut/pytest.sh`; it runs the full supported suite and can hang when failures occur.
- Skipped test files do not need investigation for this round.
- `python/tta-ut/conftest.py` and `python/tta-ut/torch_npu.py` are support shims for the pytest suite and should be treated as part of the local harness.

## Current Picture

- `python/tta-ut/pytest.sh` runs pytest from `third_party/triton-ascend` against `third_party/ascend/unittest/pytest_ut`, writes logs to `debug/tmp/pytest.log`, and does not enable MLIR dump env vars.
- `python/tta-ut/pytest_one.sh` is the single-test debug template. It currently also writes to `debug/tmp`, enables `TRITON_DEBUG=1`, and dumps passes to `debug/tmp/_pass_dump`.
- Existing debug artifacts are present in `debug/tmp/`, including `pytest.log`, `pytest.co.log`, and `pytest_one.log`.
- `third_party/triton/python/triton/backends/xyz` resolves to the repo-local `backend/` directory, so edits in `backend/compiler.py` change the backend seen by the pytest harness.
- `test_linearize_mask.py` was failing because `optimize_dynamic_offset=True` was rejected by `XYZBackend`; after adding that option to `CPUOptions`, `third_party/ascend/unittest/pytest_ut/test_linearize_mask.py` now passes all 18 cases under the `pytest_one.sh` env.
- Isolated `cann.libdevice` tests were blocked because `test_cosh.py` imports `triton.language.extra.cann.libdevice` before `torch_npu` runs; local harness now preloads `torch_npu` in `python/tta-ut/conftest.py`, and `python/tta-ut/torch_npu.py` aliases the `cann` and `ascend` libdevice imports to the shared local `triton.language.extra.libdevice` module.
- `backend/compiler.py:get_module_map()` now maps the generic, `cann`, `ascend`, and `xyz` libdevice names to `triton.language.extra.cuda.libdevice`, so isolated `test_cosh.py::test_cosh_special[float32]` emits `tt.extern_elementwise` with symbol `__nv_coshf` instead of failing on `libdevice.cosh is None`.
- `python/tta-ut/torch_npu.py` now patches `triton.compiler.code_generator.ast_to_ttir` for the local CPU harness so isolated `test_cosh.py::test_cosh_special[float32]` can build TTIR even when `tl.arange(0, 640)` lowers to `tt.make_range` with 640 elements.
- Bypassing `ast_to_ttir` verification was not enough because Triton still enforces power-of-two tensor sizes during TTIR pass-manager verification; the failing `tt.make_range` rule comes from `third_party/triton/lib/Dialect/Triton/IR/Traits.cpp`.
- `python/tta-ut/torch_npu.py` now normalizes non-power-of-two `*_SUB` constexpr launch args on the local CPU path to the highest power-of-two divisor before JIT compilation, so `test_cosh.py::test_cosh_special[float32]` no longer builds `tensor<640x...>` TTIR directly.
- Under the local harness env, isolated `third_party/ascend/unittest/pytest_ut/test_cosh.py::test_cosh_special[float32]` and `third_party/ascend/unittest/pytest_ut/test_tanh.py` now pass.
- Probing neighboring `XBLOCK_SUB` files showed `test_log2.py`, `test_sigmoid.py`, and `test_precise_div.py` already pass under the same local harness env.
- `third_party/ascend/unittest/pytest_ut/test_triton_eq.py::test_case[param_list0]` was the next real isolated failure: the CPU backend LLIR pipeline left a scalar `tt.bitcast` path from `memref<*xi1>` to `!tt.ptr<i1>` unresolved, and `mlir-translate` rejected the leftover `!tt.ptr<i1>` type.
- Adding `--triton-to-ptr` before `--convert-xyz-to-llvm` in `backend/compiler.py:make_llir()` lowers that bool-output pointer bitcast path cleanly; isolated `test_triton_eq.py`, `test_log2.py`, `test_sigmoid.py`, and `test_precise_div.py` now pass together.
- `python/tta-ut/torch_npu.py` now also normalizes bare non-power-of-two `BLOCK_SIZE` and `*_BLOCK_SIZE` constexpr launch args on the local CPU path to the highest power-of-two divisor before JIT compilation, so isolated `test_copysign.py::test_copysign[float32-shape0]` gets past the TTIR `tt.make_range` power-of-two verifier failure.
- With that bare-block normalization in place, isolated `third_party/ascend/unittest/pytest_ut/test_copysign.py::test_copysign[float32-shape0]` and `test_cyl_bessel_i0.py::test_modified_bessel_i0[param_list0]` now fail later in LLIR lowering because `one-shot-bufferize` leaves `tt.extern_elementwise` ops such as `__nv_copysignf` and `__nv_cyl_bessel_i0f` unbufferized.

## Next Move

- Investigate how the local LLIR path should lower or bufferize `tt.extern_elementwise` for libdevice-backed math ops, starting from the isolated `test_copysign.py` and `test_cyl_bessel_i0.py` failures now that the TTIR power-of-two blocker is gone.

## Risks

- Running the full `pytest.sh` suite too early can waste time or hang the session.
- Following stale path notes can hide the actual logs; the current scripts write to `debug/tmp`.
- Re-running setup can overwrite or distract from the current debug state without adding value.
- Reusing `pytest_one.sh` without changing its hardcoded target still debugs `test_abs.py`, not the current failing case.
- The pytest import aliases in `python/tta-ut/torch_npu.py` do not affect Triton codegen's separate `module_map`; keep runtime import fixes and compiler module resolution fixes distinct.
- The local constexpr normalization now covers `*_SUB`, `BLOCK_SIZE`, and `*_BLOCK_SIZE`, but tests that bake a non-power-of-two tensor shape directly into `tl.arange` can still need a different shim.
- The bool-output fix depends on the LLIR pipeline order in `backend/compiler.py`; if a future edit drops `--triton-to-ptr` again, scalar bool stores can regress at `mlir-translate` time even when TTIR and linalg lowering succeed.
- The new bare `BLOCK_SIZE` normalization only removes the front-end verifier blocker; libdevice-heavy kernels can still stop later when `tt.extern_elementwise` reaches `one-shot-bufferize` unchanged in the LLIR path.

## Rules

- Keep only what helps the next round act well.
- Update this file when reality changes.
- If code and notes disagree, trust the code and fix the notes.
