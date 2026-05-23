#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export PYTHONPATH="$ROOT/third_party/triton/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_BACKENDS_IN_TREE=1

pushd "$ROOT/third_party/FlagGems" >/dev/null
uv pip install --system scikit_build_core
uv pip install --system sqlalchemy
uv pip uninstall --system flag_gems >/dev/null 2>&1 || true
uv pip install --system --no-build-isolation -e . -v
popd >/dev/null

rm -f "$ROOT/third_party/FlagGems/tests/conftest.py"
ln -s "$ROOT/python/flag/conftest.py" \
  "$ROOT/third_party/FlagGems/tests/conftest.py"

rm -f "$ROOT/python/flag/tests"
ln -s "$ROOT/third_party/FlagGems/tests" \
  "$ROOT/python/flag/tests"
