// RUN: triton-shared-opt --split-input-file --triton-unstructured-fallback %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func public @scalar_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ARG0]], %[[CONSTANT_1]], %[[CONSTANT_0]] : !tt.ptr<f32>
// CHECK:           tt.store %[[ARG1]], %[[LOAD_0]], %[[CONSTANT_1]] : !tt.ptr<f32>
// CHECK:           tt.return
// CHECK:         }
  tt.func public @scalar_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c0 = arith.constant 0.0 : f32
    %true = arith.constant true
    %val = tt.load %src, %true, %c0 : !tt.ptr<f32>
    tt.store %dst, %val, %true : !tt.ptr<f32>
    tt.return
  }
}
