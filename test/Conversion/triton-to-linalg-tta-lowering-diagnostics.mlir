// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta="tta-pre-rewrite-tensor-pointer=false" %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @pipeline_smoke(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) {
// CHECK:           return
// CHECK:         }
  tt.func @pipeline_smoke(%arg0: !tt.ptr<f32>) {
    tt.return
  }
}
