#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRAITS_FILE="$ROOT_DIR/third_party/triton/lib/Dialect/Triton/IR/Traits.cpp"

if [[ ! -f "$TRAITS_FILE" ]]; then
  echo "missing Traits.cpp at $TRAITS_FILE" >&2
  exit 1
fi

python - "$TRAITS_FILE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

blocks = [
    """      if ((numElements & (numElements - 1)) != 0)\n        return op->emitError(\"Number of elements must be power-of-two, but \")\n               << *op << \" doesn't follow the rule (\" << numElements << \")\"\n               << \" elements\";\n""",
    """      if ((numElements & (numElements - 1)) != 0)\n        return op->emitError(\"Number of elements must be power-of-two, but \")\n               << *op << \" doesn't follow the rule (\" << numElements << \")\"\n               << \" elements\";\n""",
]

replaced = 0
for block in blocks:
    if block in text:
        text = text.replace(block, "", 1)
        replaced += 1

if replaced:
    path.write_text(text)
    print(f"patched {path} ({replaced} power-of-two checks removed)")
else:
    if "Number of elements must be power-of-two" in text:
        raise SystemExit(f"expected verifier blocks not found in {path}")
    print(f"{path} already patched")
PY

cmake --build "$ROOT_DIR/build" --target libtriton.so triton-xyz-opt -j4
