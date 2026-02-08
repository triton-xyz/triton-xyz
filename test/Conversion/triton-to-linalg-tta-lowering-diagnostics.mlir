// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --triton-to-linalg-tta="tta-pre-rewrite-tensor-pointer=false" %s
// TODO: Support boundaryCheck-based block tensor-pointer lowering in TTA route
// without requiring triton-rewrite-tensor-pointer pre-normalization.

module {
  tt.func @block_ptr_boundarycheck_unsupported(%arg0: !tt.ptr<f16>) {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
    // expected-error@+1 {{'tt.load' op failed to verify that result matches ptr type}}
    %val = tt.load %ptr {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<4x4xf16>>
    tt.store %ptr, %val {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<4x4xf16>>
    tt.return
  }
}
