# Repository Guidelines

## Project Structure & Module Organization

- `include/` and `lib/` contain the core C++ headers and implementations for the Triton Shared dialects, analyses, and conversions.
- `backend/`, `utils/`, and `misc/` hold supporting utilities and integration glue.
- `test/` contains MLIR-based regression tests organized by feature area (for example, `test/Conversion`).
- `build/` holds local build artifacts and is safe to regenerate.
- `llvm-triton/llvm-project/` contains a vendored `llvm-project` checkout. `llvm-triton/llvm-project/mlir/` is the upstream MLIR source; `llvm-triton/llvm-project/mlir/test/` is a reference for MLIR test structure and `FileCheck` style.
- `third_party/triton/` is a vendored upstream Triton checkout for reference on Triton behavior, APIs, and tests.
- `lib/Conversion/TritonToLinalg/` keeps baseline conversion passes used by `triton-to-linalg`.
- `lib/Conversion/TritonToLinalgTTA/` keeps TTA-route conversion passes used by `triton-to-linalg-tta`.
- `lib/Pipelines/TritonToLinalg.cpp` is the baseline pipeline; `lib/Pipelines/TritonToLinalgTTA.cpp` is the TTA pipeline.
- You can create a `debug_agent/` directory to store intermediate validation/testing/experimentation scripts, IR, or files instead of using `/tmp`.

## Upstream Source References

- Treat `llvm-triton/` and `third_party/triton/` as read-only reference sources unless a task explicitly requires edits there.
- For MLIR passes, dialects, or conversions, look for upstream patterns in `llvm-triton/llvm-project/mlir/` to match naming, structure, and pass/test layout.
- For new `.mlir` tests, mirror `llvm-triton/llvm-project/mlir/test/` conventions for `// RUN:` lines and `FileCheck` patterns where applicable.
- For Triton-facing changes, check `third_party/triton/` for expected behavior, API usage, and existing tests to keep parity.
- For new Triton Python kernels, use examples in `third_party/triton/python/` (especially `tutorials/`, `test/`, and `triton_kernels/`) as the primary reference.

## Build, Test, and Development Commands

- Use `utils/agent/run_kernel_case_template.sh` as the reference test harness for Triton Python kernels.
- During testing, create task-specific scripts when needed (for example under `debug_agent/`), and inspect intermediate IR dumps/logs in the corresponding output directory.
- For each kernel test, set a distinct `AGENT_DUMP_DIR` (for example `AGENT_DUMP_DIR=vec_add_case`) when running `utils/agent/run_kernel_case_template.sh` to keep IR dumps separated across runs.

- Configure and build all cmake targets.

```bash
bash utils/agent/build_cmake.sh
```

- Build `triton-xyz-opt` from the build dir.

```bash
cmake --build build --target triton-xyz-opt
```

- Quickly compare baseline vs TTA pipelines.

```bash
build/bin/triton-xyz-opt --triton-to-linalg input.mlir -o -
build/bin/triton-xyz-opt --triton-to-linalg-tta input.mlir -o -
```

- `lit -v test` runs the MLIR regression suite; narrow scope with paths like `lit -v test/Conversion`.
- When adding a lit test, refer to `utils/agent/lit_gen_demo.sh` to auto-generate `// CHECK` directives instead of writing them by hand.
- Skip `pre-commit`; handled manually.

## Triton Python Kernel Development

- Keep local kernels and runtime checks in `python/tests/` (or `python/examples/` for demos) with deterministic tensor sizes and dtypes.
- Prefer starting from a minimal kernel shape (single purpose, explicit `tl.constexpr` meta-parameters, masked memory ops for bounds safety).
- Reuse or adapt `utils/agent/run_kernel_case_template.sh` for iteration; treat it as a template that sets useful defaults (for example `TRITON_ALWAYS_COMPILE=1` and MLIR dump flags).
- Use a unique `AGENT_DUMP_DIR` per test case so `compile.log` and `triton_xyz_mlir_dump/` outputs are easy to compare and do not overwrite each other.
- Read `debug_agent/$AGENT_DUMP_DIR/compile.log` first for compiler/runtime failures, then inspect dumps in `debug_agent/$AGENT_DUMP_DIR/triton_xyz_mlir_dump/`.
- Use upstream references in `third_party/triton/python/tutorials/`, `third_party/triton/python/test/`, and `third_party/triton/python/triton_kernels/` for API patterns and expected semantics.
- When a kernel change relies on compiler transformations, add/update a focused `lit` test in `test/Conversion/` in addition to Python runtime coverage.

## Pipeline Modes (`triton-to-linalg` vs `triton-to-linalg-tta`)

- Keep both pipelines available at the same time for A/B comparison.
- `triton-to-linalg` is the baseline route and should stay close to `main` behavior.
- `triton-to-linalg-tta` is the TTA route and may evolve independently.
- Prefer adding TTA-specific passes (for example, `triton-to-tta-unstructured`) instead of changing baseline pass semantics.
- Avoid touching non-TTA passes unless the change is required by both pipelines.

## Coding Style & Naming Conventions

- C++ formatting follows LLVM style via `.clang-format`; run `clang-format` on touched C/C++ files.
- Python utilities are formatted/linted with ruff (see pre-commit hooks).
- Prefer descriptive file names that match the existing pattern (for example, `TritonToStructuredPass.cpp`).

## Testing Guidelines

- Tests are `.mlir` files run by `lit` and verified with `FileCheck` in `// RUN:` lines.
- Add new tests under the closest feature area (for example, `test/Conversion/TritonToPtr`).
- Baseline behavior tests should use `--triton-to-linalg` and baseline pass names.
- TTA behavior tests should use `--triton-to-linalg-tta` or `--triton-to-tta-*`.
- Keep baseline and TTA expectations in separate test files or split-input sections; avoid mixing unrelated routes in one check flow.
- Prefer grouping related cases that exercise the same pass in a single file; avoid mixing unrelated features.
- Keep each case minimal and use focused `CHECK:` patterns to avoid over-specifying behavior.
- When adding or modifying tests with `FileCheck`, regenerate check lines by following `utils/agent/lit_gen_demo.sh` (use it as the single source of truth, and avoid hand-editing large `CHECK` blocks).
- After regeneration, run targeted validation first (for touched files) and then broader `lit`/`lit -v` as needed.

## Commit & Pull Request Guidelines

- Recent commit messages are short, imperative, and mostly lowercase (for example, `sort includes`, `lit test`).
- Keep commits focused; avoid mixing formatting with functional changes unless required.
- In PRs, describe the behavioral change, link related issues, and include test commands or output when applicable.
