# Repository Guidelines

## Project Structure & Module Organization

- `include/` and `lib/` contain the core C++ headers and implementations for the Triton Shared dialects, analyses, and conversions.
- `backend/`, `utils/`, and `misc/` hold supporting utilities and integration glue.
- `test/` contains MLIR-based regression tests organized by feature area (for example, `test/Conversion`).
- `build/` holds local build artifacts and is safe to regenerate.
- `llvm-triton/llvm-project/` contains a vendored `llvm-project` checkout. `llvm-triton/llvm-project/mlir/` is the upstream MLIR source; `llvm-triton/llvm-project/mlir/test/` is a reference for MLIR test structure and `FileCheck` style.
- `third_party/triton/` is a vendored upstream Triton checkout for reference on Triton behavior, APIs, and tests.
- You can create a `debug_agent/` directory to store intermediate validation/testing/experimentation scripts, IR, or files instead of using `/tmp`.

## Upstream Source References

- Treat `llvm-triton/` and `third_party/triton/` as read-only reference sources unless a task explicitly requires edits there.
- For MLIR passes, dialects, or conversions, look for upstream patterns in `llvm-triton/llvm-project/mlir/` to match naming, structure, and pass/test layout.
- For new `.mlir` tests, mirror `llvm-triton/llvm-project/mlir/test/` conventions for `// RUN:` lines and `FileCheck` patterns where applicable.
- For Triton-facing changes, check `third_party/triton/` for expected behavior, API usage, and existing tests to keep parity.

## Build, Test, and Development Commands

- Use CMake only; skip Python setup.
- Configure and build all cmake targets.

```bash
bash utils/agent/build_cmake.sh
```

- Build `triton-xyz-opt` from the build dir.

```bash
cmake --build build --target triton-xyz-opt
```

- `lit -v test` runs the MLIR regression suite; narrow scope with paths like `lit -v test/Conversion`.
- When adding a lit test, refer to `utils/agent/lit_gen_demo.sh` to auto-generate `// CHECK` directives instead of writing them by hand.
- Skip `pre-commit`; handled manually.

## Coding Style & Naming Conventions

- C++ formatting follows LLVM style via `.clang-format`; run `clang-format` on touched C/C++ files.
- Python utilities are formatted/linted with ruff (see pre-commit hooks).
- Prefer descriptive file names that match the existing pattern (for example, `TritonToStructuredPass.cpp`).

## Testing Guidelines

- Tests are `.mlir` files run by `lit` and verified with `FileCheck` in `// RUN:` lines.
- Add new tests under the closest feature area (for example, `test/Conversion/TritonToPtr`).
- Prefer grouping related cases that exercise the same pass in a single file; avoid mixing unrelated features.
- Keep each case minimal and use focused `CHECK:` patterns to avoid over-specifying behavior.

## Commit & Pull Request Guidelines

- Recent commit messages are short, imperative, and mostly lowercase (for example, `sort includes`, `lit test`).
- Keep commits focused; avoid mixing formatting with functional changes unless required.
- In PRs, describe the behavioral change, link related issues, and include test commands or output when applicable.
