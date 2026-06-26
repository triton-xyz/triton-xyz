// RUN: triton-xyz-opt %s --triton-to-xyz='unstructured-offset-bit-width=64 assert-to-cf=false pids-to-func-args=true' -dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s

// CHECK: triton-to-tta-unstructured{offset-bit-width=64}
// CHECK: triton-pids-to-func-args
// CHECK: triton-arith-to-linalg{assert-to-cf=false}

module {
  tt.func @empty() {
    tt.return
  }
}
