// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta="tta-pre-rewrite-tensor-pointer=false" %s | FileCheck %s
// TODO: Support boundaryCheck-based block tensor-pointer lowering in TTA route
// without requiring triton-rewrite-tensor-pointer pre-normalization.

module {
// CHECK-LABEL:   func.func @block_ptr_boundarycheck_unsupported(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : i64
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<*xf16> to !tt.ptr<f16>
// CHECK:           %[[MAKE_TENSOR_PTR_0:.*]] = tt.make_tensor_ptr %[[UNREALIZED_CONVERSION_CAST_0]], {{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]], {{\[}}%[[CONSTANT_2]], %[[CONSTANT_1]]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "tensor_ptr_unhandled"} : <tensor<4x4xf16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[MAKE_TENSOR_PTR_0]] {boundaryCheck = array<i32: 1>, padding = 2 : i32, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"} : !tt.ptr<tensor<4x4xf16>>
// CHECK:           tt.store %[[MAKE_TENSOR_PTR_0]], %[[LOAD_0]] {boundaryCheck = array<i32: 1>, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"} : !tt.ptr<tensor<4x4xf16>>
// CHECK:           return
// CHECK:         }
  tt.func @block_ptr_boundarycheck_unsupported(%arg0: !tt.ptr<f16>) {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
    %val = tt.load %ptr {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<4x4xf16>>
    tt.store %ptr, %val {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<4x4xf16>>
    tt.return
  }
}
