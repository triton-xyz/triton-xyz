// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta="tta-pre-rewrite-tensor-pointer=false" %s | FileCheck %s
// TODO: Support boundaryCheck-based block tensor-pointer lowering in TTA route
// without requiring triton-rewrite-tensor-pointer pre-normalization.

module {
  // CHECK-LABEL: func.func @block_ptr_boundarycheck_unsupported(
  // CHECK: tt.make_tensor_ptr {{.*}} {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "tensor_ptr_unhandled"}
  // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 2 : i32, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"} : !tt.ptr<tensor<4x4xf16>>
  // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"}
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
