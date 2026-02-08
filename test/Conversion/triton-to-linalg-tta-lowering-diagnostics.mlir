// RUN: triton-xyz-opt --verify-diagnostics --triton-to-linalg-tta %s

// this can pass
// triton-xyz-opt --triton-rewrite-tensor-pointer --triton-to-linalg-tta %s
// TODO: Support block tensor-pointer boundary semantics in TTA lowering even
// when triton-rewrite-tensor-pointer is not run beforehand.

module {
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
