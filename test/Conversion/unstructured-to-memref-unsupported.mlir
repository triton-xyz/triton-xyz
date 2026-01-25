// RUN: triton-shared-opt --split-input-file --verify-diagnostics --unstructured-to-memref %s

// TODO: emit message before legalization

module {
  tt.func public @scalar_offset_tensor_gather(%src: !tt.ptr<f32>, %off: i32) {
    // expected-error@+1 {{failed to legalize operation 'tts.gather' that was explicitly marked illegal}}
    %val = tts.gather %src[%off] : (<f32>, i32) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
  tt.func public @scalar_offset_tensor_scatter(%dst: !tt.ptr<f32>, %off: i32) {
    %vals = arith.constant dense<1.000000e+00> : tensor<4xf32>
    // expected-error@+1 {{failed to legalize operation 'tts.scatter' that was explicitly marked illegal}}
    tts.scatter %vals into %dst[%off] : tensor<4xf32> into (<f32>, i32)
    tt.return
  }
}
