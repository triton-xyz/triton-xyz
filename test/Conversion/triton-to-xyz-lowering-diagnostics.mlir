// RUN: triton-xyz-opt --split-input-file --triton-to-xyz %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @pipeline_smoke(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: memref<*xf32>) {
// CHECK:           return
// CHECK:         }
  tt.func @pipeline_smoke(%arg0: !tt.ptr<f32>) {
    tt.return
  }
}
