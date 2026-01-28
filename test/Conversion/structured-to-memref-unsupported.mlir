// RUN: triton-xyz-opt --verify-diagnostics --structured-to-memref %s

module {
  tt.func @unsupported_order(%base: !tt.ptr<f32>) {
    // expected-error@+2 {{non-decreasing dimension order on tensor pointers are not yet supported}}
    // expected-error@+1 {{failed to legalize operation 'tts.make_tptr' that was explicitly marked illegal}}
    %tptr = tts.make_tptr %base to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [2, 2], order: [0, 1] : <f32> to !tt.ptr<tensor<2x2xf32>>
    %val = "tts.load"(%tptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<2x2xf32>>) -> tensor<2x2xf32>
    tt.return
  }
}
