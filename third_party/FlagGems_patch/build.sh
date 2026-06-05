#!/usr/bin/env bash
set -euo pipefail

rm -f "$PWD/third_party/FlagGems/patch.patch"
ln -s "$PWD/third_party/FlagGems_patch/patch.patch" "$PWD/third_party/FlagGems/"

rm -f "$PWD/third_party/FlagGems/src/flag_gems/runtime/backend/_xyz"
ln -s "$PWD/third_party/FlagGems_patch/_xyz" \
  "$PWD/third_party/FlagGems/src/flag_gems/runtime/backend/"

pushd third_party/FlagGems
if git apply --check patch.patch >/dev/null 2>&1; then
  git apply patch.patch
else
  git apply --reverse --check patch.patch >/dev/null 2>&1
fi

uv pip uninstall --system flag_gems
uv pip install --system --no-build-isolation -e . -v
popd

rm -f "${PWD}/third_party/FlagGems/tests/conftest.py"
ln -s "${PWD}/python/flag/conftest.py" \
  "${PWD}/third_party/FlagGems/tests/conftest.py"

rm -f "${PWD}/third_party/FlagGems/tests/torch_xyz.py"
ln -s "${PWD}/python/flag/torch_xyz.py" \
  "${PWD}/third_party/FlagGems/tests/torch_xyz.py"
