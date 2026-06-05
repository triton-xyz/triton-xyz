#!/usr/bin/env bash
set -euo pipefail

if [[ ! -e "$PWD/third_party/FlagGems/patch.patch" ]]; then
  ln -s $PWD/third_party/FlagGems_patch/patch.patch $PWD/third_party/FlagGems/

  pushd $PWD/third_party/FlagGems
  git apply patch.patch
  popd
fi

pushd third_party/FlagGems
uv pip uninstall --system flag_gems
uv pip install --system --no-build-isolation -e . -v
popd

rm -f "${PWD}/third_party/FlagGems/tests/conftest.py"
ln -s "${PWD}/python/flag/conftest.py" \
  "${PWD}/third_party/FlagGems/tests/conftest.py"

rm -f "${PWD}/third_party/FlagGems/tests/torch_xyz.py"
ln -s "${PWD}/python/flag/torch_xyz.py" \
  "${PWD}/third_party/FlagGems/tests/torch_xyz.py"
