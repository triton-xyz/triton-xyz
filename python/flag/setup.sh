#!/usr/bin/env bash
set -euo pipefail

pushd third_party/FlagGems
uv pip uninstall --system flag_gems
uv pip install --system --no-build-isolation -e . -v
popd

rm -f "${PWD}/third_party/FlagGems/tests/conftest.py"
ln -s "${PWD}/python/flag/conftest.py" \
  "${PWD}/third_party/FlagGems/tests/conftest.py"

rm -f "${PWD}/python/flag/tests"
ln -s "${PWD}/third_party/FlagGems/tests" \
  "${PWD}/python/flag/tests"
