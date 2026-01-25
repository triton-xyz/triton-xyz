#!/usr/bin/env bash
set -euo pipefail

# get cmd from lit test, only support basic style

# usage:
# utils/lit_run_cmd.sh test/Conversion/triton-to-structured-prepass.mlir

file="${1:?usage: $0 <test.mlir>}"
rg '^[[:space:]]*//[[:space:]]*RUN:' "$file" |
  sed -E 's@^[[:space:]]*//[[:space:]]*RUN:[[:space:]]*@@' |
  while IFS= read -r run; do
    cmd="${run//%s/$file}"
    echo "$cmd"
  done
