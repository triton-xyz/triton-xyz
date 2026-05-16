# triton-xyz

`triton-xyz` keeps the retained Triton-to-Linalg path centered on the TTA route. For new lowering work, prefer `triton-to-linalg-tta` and the TTA-specific passes under `lib/Conversion/TritonToLinalgTTA/`.

## Build

Configure and build with the repo helper:

```bash
bash utils/agent/build_cmake.sh
```

Build the main optimizer directly from the build tree when iterating on passes:

```bash
cmake --build build --target triton-xyz-opt
```

## Test

Run the MLIR regression suite with `lit`:

```bash
lit -v test
```

Narrow the scope when iterating on a pass or dialect:

```bash
lit -v test/Conversion
lit -v test/Dialect/triton-address-dialect.mlir
```

The retained full lowering route is:

```bash
build/bin/triton-xyz-opt --triton-to-linalg-tta input.mlir -o -
```

## Python Harnesses

Use the provided helpers in `utils/agent/` so runtime tests run with a consistent dump and environment setup.

Run a single Python kernel script:

```bash
AGENT_DUMP_DIR=vec_add_case \
KERNEL_PY=python/tests/test_vec_add.py \
bash utils/agent/run_kernel_case_template.sh
```

Run a specific pytest case:

```bash
AGENT_DUMP_DIR=structured_mask_3 \
PYTEST_TARGET='python/tests/test_triton_to_structured.py::test_masked_1d[3]' \
bash utils/agent/run_pytest_case_template.sh -q
```

Run the current plan-29 Python coverage matrix:

```bash
bash utils/agent/run_plan29_python_matrix.sh
```

## Dump Directories

All helper scripts write dumps under `debug_agent/$AGENT_DUMP_DIR/` by default. Each run gets:

- `compile.log` or `pytest.log`
- `triton_dump/`
- `triton_xyz_mlir_dump/`

Use a distinct `AGENT_DUMP_DIR` per case so outputs do not overwrite each other:

```bash
AGENT_DUMP_DIR=masked_2d_case \
KERNEL_PY=python/tests/test_triton_to_unstructured.py \
bash utils/agent/run_kernel_case_template.sh
```

When debugging a failing kernel, read `debug_agent/$AGENT_DUMP_DIR/compile.log` first, then inspect `triton_xyz_mlir_dump/`.
