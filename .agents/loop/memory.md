# Memory

Use this file for short, reusable facts worth carrying across rounds.

## Facts

- The TTA pytest work is rooted in `python/tta-ut/` and targets `third_party/triton-ascend/third_party/ascend/unittest/pytest_ut`.
- `python/tta-ut/setup.sh` clones `third_party/triton-ascend` at commit `115f51f71917e6836c3138a8f0d52fb71caf1d63` and symlinks local `conftest.py` and `torch_npu.py` into the vendored pytest directory.
- `python/tta-ut/conftest.py` filters and reshapes the upstream pytest matrix; skipped test files listed there are intentionally out of scope.
- `python/tta-ut/torch_npu.py` provides the local compatibility shim so the Ascend-oriented tests can run on the current XYZ backend and CPU-based torch environment.

## Learnings

- `python/tta-ut/pytest.sh` writes its artifacts to `debug/tmp/` and tees the main run log to `debug/tmp/pytest.log`.
- `python/tta-ut/pytest_one.sh` is the intended single-test debug entry point; it enables MLIR dump env vars and currently writes both logs and pass dumps under `debug/tmp/`.
- The repository currently contains prior debug logs in `debug/tmp/`: `pytest.log`, `pytest.co.log`, and `pytest_one.log`.

## Open Questions

- Which specific non-skipped pytest case is the best first reproducer for the current compiler/runtime failure?
- Does `python/tta-ut/pytest_one.sh` need to be updated to use an isolated output directory such as `debug/tmp-pytest_one`, or should the workflow standardize on `debug/tmp`?

## Avoid Repeating

- Do not rerun `python/tta-ut/setup.sh` unless the vendored checkout or symlinks are known to be broken.
- Do not start with `bash python/tta-ut/pytest.sh` for debugging; narrow with `pytest_one.sh` first to avoid hangs and noisy logs.

## Stale

- Historical note said an interrupted full-suite log lived at `debug/tmp-0/pytest.log`; current repo state has `debug/tmp/pytest.log` instead.
- Historical note said `pytest_one.sh` dumps to `debug/tmp-pytest_one`; current script writes to `debug/tmp`.

## Rules

- Keep only reusable information.
- Summarize outcomes. Do not paste raw logs or long transcripts.
- Prefer concrete evidence such as file paths, commands, or commit hashes.
