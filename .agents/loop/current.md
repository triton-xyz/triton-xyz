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

## Next Move

- Use `python/tta-ut/pytest_one.sh` as the template for narrowing to one failing test, then inspect `debug/tmp/pytest_one.log` and `debug/tmp/_pass_dump/` to isolate the compiler failure.

## Risks

- Running the full `pytest.sh` suite too early can waste time or hang the session.
- Following stale path notes can hide the actual logs; the current scripts write to `debug/tmp`.
- Re-running setup can overwrite or distract from the current debug state without adding value.

## Rules

- Keep only what helps the next round act well.
- Update this file when reality changes.
- If code and notes disagree, trust the code and fix the notes.
