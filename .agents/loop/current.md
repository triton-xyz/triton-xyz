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
- The latest isolated `pytest` run still fails earlier than expected during AST to TTIR parsing: `tl.arange(0, 640)` becomes `tt.make_range` with 640 elements, and the active frontend path still rejects that non-power-of-two span.

## Next Move

- Fix the real frontend path used by isolated `test_cosh.py` so non-power-of-two `tl.arange` and block shapes are accepted during AST to TTIR parsing, then rerun `third_party/ascend/unittest/pytest_ut/test_cosh.py::test_cosh_special[float32]`.

## Risks

- Running the full `pytest.sh` suite too early can waste time or hang the session.
- Following stale path notes can hide the actual logs; the current scripts write to `debug/tmp`.
- Re-running setup can overwrite or distract from the current debug state without adding value.
- Reusing `pytest_one.sh` without changing its hardcoded target still debugs `test_abs.py`, not the current failing case.
- The pytest import aliases in `python/tta-ut/torch_npu.py` do not affect Triton codegen's separate `module_map`; keep runtime import fixes and compiler module resolution fixes distinct.
- Earlier notes claiming `test_cosh.py` already got past `tl.arange(0, 640)` are stale; trust the latest isolated `pytest` run over older harness assumptions.

## Rules

- Keep only what helps the next round act well.
- Update this file when reality changes.
- If code and notes disagree, trust the code and fix the notes.
