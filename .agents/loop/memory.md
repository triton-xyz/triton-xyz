# Memory

Use this file for short, reusable facts worth carrying across rounds.

## Facts

- The TTA pytest work is rooted in `python/tta-ut/` and targets `third_party/triton-ascend/third_party/ascend/unittest/pytest_ut`.
- `python/tta-ut/setup.sh` clones `third_party/triton-ascend` at commit `115f51f71917e6836c3138a8f0d52fb71caf1d63` and symlinks local `conftest.py` and `torch_npu.py` into the vendored pytest directory.
- `python/tta-ut/conftest.py` filters and reshapes the upstream pytest matrix; skipped test files listed there are intentionally out of scope.
- `python/tta-ut/torch_npu.py` provides the local compatibility shim so the Ascend-oriented tests can run on the current XYZ backend and CPU-based torch environment.
- `third_party/triton/python/triton/backends/xyz` is a symlink to the repo-local `backend/` directory.

## Learnings

- `python/tta-ut/pytest.sh` writes its artifacts to `debug/tmp/` and tees the main run log to `debug/tmp/pytest.log`.
- `python/tta-ut/pytest_one.sh` is the intended single-test debug entry point; it enables MLIR dump env vars and currently writes both logs and pass dumps under `debug/tmp/`.
- The repository currently contains prior debug logs in `debug/tmp/`: `pytest.log`, `pytest.co.log`, and `pytest_one.log`.
- In this shell, TTA pytest runs need `PYTHONPATH=$PWD/third_party/triton/python`; without it, `import triton` fails.
- `third_party/ascend/unittest/pytest_ut/test_linearize_mask.py` passes once `backend/compiler.py` accepts the `optimize_dynamic_offset` JIT kwarg via `CPUOptions`.
- Isolated `cann.libdevice` pytest cases need `torch_npu` loaded during collection; importing it from `python/tta-ut/conftest.py` installs the local module aliases early enough for `import triton.language.extra.cann.libdevice`.
- `python/tta-ut/torch_npu.py` must alias `triton.language.extra.cann.libdevice` and `triton.language.extra.ascend.libdevice` to `third_party/triton/python/triton/language/extra/libdevice.py`, not the placeholder `extra/xyz/libdevice.py`.
- `backend/compiler.py:get_module_map()` can point the generic, `cann`, `ascend`, and `xyz` libdevice names at `triton.language.extra.cuda.libdevice`; that makes `libdevice.cosh` lower to `tt.extern_elementwise` with symbol `__nv_coshf`.
- `python/tta-ut/torch_npu.py` can patch `triton.compiler.code_generator.ast_to_ttir` for the local CPU harness to skip the first TTIR `module.verify()` gate and expose later failures.
- A `backend/compiler.py:make_ttir()` verifier bypass is the wrong next move for this class of failure: Triton itself enforces power-of-two tensor sizes in `third_party/triton/lib/Dialect/Triton/IR/Traits.cpp`, and the TTIR pass pipeline verifies that invariant.
- `python/tta-ut/torch_npu.py` can normalize non-power-of-two `*_SUB` constexpr launch args on the local CPU path to the highest power-of-two divisor before JIT compilation; that turns cases like `XBLOCK_SUB=640` into verifier-friendly chunks (`128` here) without editing vendored tests.
- With that launch-time normalization in place, isolated `third_party/ascend/unittest/pytest_ut/test_cosh.py::test_cosh_special[float32]` and `third_party/ascend/unittest/pytest_ut/test_tanh.py` pass under the local harness env.

## Open Questions

- Should the non-power-of-two `tl.arange` and block-shape relaxation stay as a pytest-harness-only shim, or should the same behavior move into the shared Triton frontend/backend path?
- Does `python/tta-ut/pytest_one.sh` need a parameterized target instead of the current hardcoded `test_abs.py` entry?

## Avoid Repeating

- Do not rerun `python/tta-ut/setup.sh` unless the vendored checkout or symlinks are known to be broken.
- Do not start with `bash python/tta-ut/pytest.sh` for debugging; narrow with `pytest_one.sh` first to avoid hangs and noisy logs.

## Stale

- Historical note said an interrupted full-suite log lived at `debug/tmp-0/pytest.log`; current repo state has `debug/tmp/pytest.log` instead.
- Historical note said `pytest_one.sh` dumps to `debug/tmp-pytest_one`; current script writes to `debug/tmp`.
- Historical note said the next decision was whether to bypass `backend/compiler.py:make_ttir()` verification; current evidence shows the durable fix direction is launch-time sub-block normalization instead.

## Rules

- Keep only reusable information.
- Summarize outcomes. Do not paste raw logs or long transcripts.
- Prefer concrete evidence such as file paths, commands, or commit hashes.
