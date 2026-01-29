// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --normalize-tensor-ptr-order %s

module {
  tt.func @bad_order(%base: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    // expected-error@+1 {{order is not a permutation when normalizing}}
    %tptr = tts.make_tptr %base to sizes: [4, 8], strides: [%c8, %c1], offsets: [%c0, %c0], shape: [%c4, %c8], order: [0, 2] : <f16> to !tt.ptr<tensor<4x8xf16>>
    %val = "tts.load"(%tptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x8xf16>>) -> tensor<4x8xf16>
    tt.return
  }
}

// -----

module {
  func.func private @use(%p: !tt.ptr<tensor<4x8xf16>>)
  tt.func @unsupported_user(%base: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %tptr = tts.make_tptr %base to sizes: [4, 8], strides: [%c8, %c1], offsets: [%c0, %c0], shape: [%c4, %c8], order: [0, 1] : <f16> to !tt.ptr<tensor<4x8xf16>>
    // expected-error@+1 {{unsupported user when normalizing tts.make_tptr order}}
    func.call @use(%tptr) : (!tt.ptr<tensor<4x8xf16>>) -> ()
    tt.return
  }
}

// -----

module {
  tt.func @mask_rank_mismatch(%base: !tt.ptr<f16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %other = arith.constant 0.0 : f16
    %mask = arith.constant 4 : index
    %tptr = tts.make_tptr %base to sizes: [4, 8], strides: [%c8, %c1], offsets: [%c0, %c0], shape: [%c4, %c8], order: [0, 1] : <f16> to !tt.ptr<tensor<4x8xf16>>
    // expected-error@+1 {{mask rank mismatch when normalizing order}}
    %val = "tts.load"(%tptr, %mask, %other) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (!tt.ptr<tensor<4x8xf16>>, index, f16) -> tensor<4x8xf16>
    tt.return
  }
}
