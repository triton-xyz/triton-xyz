// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --tta-to-memref %s

module {
  tt.func @non_zero_indirect_offset(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %idx = "tta.reindex"(%addr, %offsets) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 1>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<4x!tt.ptr<f32>>
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @indirect_on_block_ptr(%base: !tt.ptr<f16>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr = tta.make_addr %base to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], shape: [4, 4], order: [1, 0] : <f16> to !tt.ptr<tensor<4x4xf16>>
    %idx = "tta.reindex"(%addr, %offsets) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (!tt.ptr<tensor<4x4xf16>>, tensor<4xi32>) -> !tt.ptr<tensor<4x4xf16>>
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x4xf16>>) -> tensor<4x4xf16>
    tt.return
  }
}
