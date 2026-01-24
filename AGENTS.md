# Repository Guidelines

## Project Structure & Module Organization

- `include/` and `lib/` contain the core C++ headers and implementations for the Triton Shared dialects, analyses, and conversions.
- `backend/`, `utils/`, and `misc/` hold supporting utilities and integration glue.
- `test/` contains MLIR-based regression tests organized by feature area (for example, `test/Conversion`).
- `llvm-triton/` and `triton/` are upstream dependencies; treat them as vendored sources.
- `build/` is for local build artifacts and is safe to regenerate.

## Build, Test, and Development Commands

- Use CMake only; skip Python setup.
- Configure and build all cmake targets.

```bash
[[ "$(uname)" == "Darwin" ]] && PRESET="osx_lld" || PRESET="osx"
cmake --preset $PRESET -S$PWD/triton -B$PWD/build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
  -DTRITON_PLUGIN_DIRS="$PWD" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/build/bin" \
  -DTRITON_WHEEL_DIR="$PWD/build/bin" \
  -DPython3_EXECUTABLE=$(which python)
cmake --build $PWD/build --target all
```

- Build `triton-shared-opt` from the build dir.

```bash
cmake --build build --target triton-shared-opt
```

- `lit -v test` runs the MLIR regression suite; narrow scope with paths like `lit -v test/Conversion`.
- skip `pre-commit`, let human do it

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
