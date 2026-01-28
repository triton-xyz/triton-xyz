// RUN: triton-xyz-opt --split-input-file --triton-ptr-to-memref %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @func_ptr_args(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xi8>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           return
// CHECK:         }
  func.func @func_ptr_args(%arg0: !tt.ptr<f32>, %arg1: tensor<4x!tt.ptr<i8>>, %arg2: i32) {
    return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tt_ptr_arg(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf16>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<*xf16> to !tt.ptr<f16>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:           tt.store %[[UNREALIZED_CONVERSION_CAST_0]], %[[CONSTANT_0]] : !tt.ptr<f16>
// CHECK:           tt.return
// CHECK:         }
  tt.func @tt_ptr_arg(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant 1.000000e+00 : f16
    tt.store %arg0, %cst : !tt.ptr<f16>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   func.func @callee(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           return %[[CONSTANT_0]] : f32
// CHECK:         }
  func.func @callee(%arg0: !tt.ptr<f32>) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    return %cst : f32
  }

// CHECK-LABEL:   func.func @caller(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) -> f32 {
// CHECK:           %[[VAL_0:.*]] = call @callee(%[[ARG0]]) : (memref<*xf32>) -> f32
// CHECK:           return %[[VAL_0]] : f32
// CHECK:         }
  func.func @caller(%arg0: !tt.ptr<f32>) -> f32 {
    %0 = func.call @callee(%arg0) : (!tt.ptr<f32>) -> f32
    return %0 : f32
  }
}
