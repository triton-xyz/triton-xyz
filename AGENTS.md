# Repository Guidelines

## Project Structure & Module Organization

- `include/` and `lib/` contain the core C++ headers and implementations for the Triton Shared dialects, analyses, and conversions.
- `backend/`, `utils/`, and `misc/` hold supporting utilities and integration glue.
- `test/` contains MLIR-based regression tests organized by feature area (for example, `test/Conversion`).
- `build/` holds local build artifacts and is safe to regenerate.
- `llvm-triton/llvm-project/` contains a vendored `llvm-project` checkout. `llvm-triton/llvm-project/mlir/` is the upstream MLIR source; `llvm-triton/llvm-project/mlir/test/` is a reference for MLIR test structure and `FileCheck` style.
- `third_party/triton/` is a vendored upstream Triton checkout for reference on Triton behavior, APIs, and tests.

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

- Build `triton-shared-opt` from the build dir.

```bash
cmake --build build --target triton-shared-opt
```

- `lit -v test` runs the MLIR regression suite; narrow scope with paths like `lit -v test/Conversion`.
- Skip `pre-commit`; handled manually.

## Coding Style & Naming Conventions

- C++ formatting follows LLVM style via `.clang-format`; run `clang-format` on touched C/C++ files.
- Python utilities are formatted/linted with ruff (see pre-commit hooks).
- Prefer descriptive file names that match the existing pattern (for example, `TritonToStructuredPass.cpp`).

## Testing Guidelines

- Tests are `.mlir` files executed by `lit` and checked with `FileCheck` directives in `// RUN:` lines.
- Add new tests alongside the closest existing feature area (for example, `test/Conversion/TritonToPtr`).
- Keep tests minimal: one feature per file when possible, with focused `CHECK:` patterns.

## Commit & Pull Request Guidelines

- Recent commit messages are short, imperative, and mostly lowercase (for example, `sort includes`, `lit test`).
- Keep commits focused; avoid mixing formatting with functional changes unless required.
- In PRs, describe the behavioral change, link related issues, and include test commands or output when applicable.
