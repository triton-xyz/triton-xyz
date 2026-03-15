// RUN: triton-xyz-opt --split-input-file %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func public @nop_1d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<16xf32>) -> tensor<16xf32> {
// CHECK:           %[[NOP_0:.*]] = tt.nop %[[ARG0]] : tensor<16xf32>
// CHECK:           tt.return %[[NOP_0]] : tensor<16xf32>
// CHECK:         }
  tt.func public @nop_1d(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = tt.nop %arg0 : tensor<16xf32>
    tt.return %0 : tensor<16xf32>
  }
}
