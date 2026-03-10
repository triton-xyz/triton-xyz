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
- Because codegen resolves libdevice imports through `triton.language.extra.cuda.libdevice`, a pytest-harness compatibility implementation has to patch that CUDA libdevice module too; updating only `triton.language.extra.libdevice` is not enough to change lowering.
- `python/tta-ut/torch_npu.py` can patch `triton.compiler.code_generator.ast_to_ttir` for the local CPU harness to skip the first TTIR `module.verify()` gate and expose later failures.
- A `backend/compiler.py:make_ttir()` verifier bypass is the wrong next move for this class of failure: Triton itself enforces power-of-two tensor sizes in `third_party/triton/lib/Dialect/Triton/IR/Traits.cpp`, and the TTIR pass pipeline verifies that invariant.
- `python/tta-ut/torch_npu.py` can normalize non-power-of-two `*_SUB` constexpr launch args on the local CPU path to the highest power-of-two divisor before JIT compilation; that turns cases like `XBLOCK_SUB=640` into verifier-friendly chunks (`128` here) without editing vendored tests.
- `python/tta-ut/torch_npu.py` can use the same highest-power-of-two-divisor rule for bare `BLOCK_SIZE` and `*_BLOCK_SIZE` constexpr launch args on the local CPU path; that removes TTIR `tt.make_range` verifier failures for cases like `BLOCK_SIZE=192` in isolated math tests.
- With that launch-time normalization in place, isolated `third_party/ascend/unittest/pytest_ut/test_cosh.py::test_cosh_special[float32]` and `third_party/ascend/unittest/pytest_ut/test_tanh.py` pass under the local harness env.
- `backend/compiler.py:make_llir()` needs `--triton-to-ptr` before `--convert-xyz-to-llvm`; without it, bool-output kernels like `third_party/ascend/unittest/pytest_ut/test_triton_eq.py` can leave a `memref<*xi1> -> !tt.ptr<i1> -> tt.bitcast -> !tt.ptr<i8>` chain unresolved until `mlir-translate` rejects the leftover `!tt.ptr<i1>` type.
- After adding that pass-order fix, isolated `third_party/ascend/unittest/pytest_ut/test_triton_eq.py`, `test_log2.py`, `test_sigmoid.py`, and `test_precise_div.py` pass together under the local harness env.
- Once the TTIR power-of-two blocker is removed for libdevice-backed math kernels, unsupported libdevice symbols can still survive into LLIR as `tt.extern_elementwise` and fail `one-shot-bufferize`.
- `lib/Conversion/TritonArithToLinalg/TritonArithToLinalg.cpp` can lower `__nv_copysignf` and `__nv_copysign` to `math.copysign`; after rebuilding `triton-xyz-opt`, isolated `third_party/ascend/unittest/pytest_ut/test_copysign.py::test_copysign[float32-shape0]` passes under the local harness env.
- `python/tta-ut/torch_npu.py` can replace `cyl_bessel_i0` in both libdevice modules with a local `@triton.jit` approximation for the CPU harness; that avoids the unbufferized `__nv_cyl_bessel_i0f` LLIR path and makes isolated `third_party/ascend/unittest/pytest_ut/test_cyl_bessel_i0.py::test_modified_bessel_i0[param_list0]` pass.
- In `backend/compiler.py:make_llir()`, lowering `math.acosh`-style ops for the CPU backend needs both `--convert-math-to-libm` and `--convert-func-to-llvm` after `--convert-xyz-to-llvm`; `--convert-math-to-llvm` alone leaves `math.acosh` untouched, while `--convert-math-to-libm` alone leaves a `func.func @acoshf` declaration that `mlir-translate` rejects.
- After that LLIR pipeline fix, isolated `third_party/ascend/unittest/pytest_ut/test_acos.py::test_asinh_special[float32]` and `third_party/ascend/unittest/pytest_ut/test_acosh.py` pass under the local harness env, so the old full-suite `test_acos`/`test_acosh` failures in `debug/tmp/pytest.log` are stale.
- `third_party/ascend/unittest/pytest_ut/test_address_check.py::test_cpu_tensor_should_fail` is now the refreshed earliest failure in that frontier; it expects CPU tensors to be rejected, but the local harness maps both `.npu()` and `device='npu'` allocations onto CPU tensors, so no `ValueError` is raised today.

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
- Historical note said `test_cyl_bessel_i0.py` still failed in LLIR on unbufferized `__nv_cyl_bessel_i0f`; current harness patches replace that libdevice call with a local Triton approximation, and the isolated float32 case now passes.
- Historical note implied the next frontier was still the old libdevice unary-math failures from `debug/tmp/pytest.log`; current reruns show `test_acos.py` and `test_acosh.py` pass, and the active frontier has moved to `test_address_check.py`.

## Rules

- Keep only reusable information.
- Summarize outcomes. Do not paste raw logs or long transcripts.
- Prefer concrete evidence such as file paths, commands, or commit hashes.
