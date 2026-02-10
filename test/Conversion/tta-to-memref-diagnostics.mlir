// RUN: triton-xyz-opt --split-input-file --verify-diagnostics --tta-to-memref %s

module {
  tt.func @chained_indirect_mixed_dims(%src: !tt.ptr<f32>) {
    %offsets0 = arith.constant dense<[0, 1]> : tensor<2xi32>
    %offsets1 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x4x!tt.ptr<f32>>
    %idx0 = "tta.reindex"(%addr, %offsets0) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (tensor<2x4x!tt.ptr<f32>>, tensor<2xi32>) -> tensor<2x4x!tt.ptr<f32>>
    %idx1 = "tta.reindex"(%idx0, %offsets1) <{indirect_dim = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (tensor<2x4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<2x4x!tt.ptr<f32>>
    // expected-error@+2 {{tta-to-memref: mixed_indirect_dim in address chain}}
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%idx1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x4x!tt.ptr<f32>>) -> tensor<2x4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @chained_indirect_index_not_mergeable(%src: !tt.ptr<f32>) {
    %offsets_a = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %offsets_b = arith.constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
    %mask_a = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %mask_b = arith.constant dense<[true, false, true, false, true]> : tensor<5xi1>
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %idx0 = "tta.reindex"(%addr, %offsets_a, %mask_a) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4x!tt.ptr<f32>>
    %idx1 = "tta.reindex"(%idx0, %offsets_b, %mask_b) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (tensor<4x!tt.ptr<f32>>, tensor<5xi32>, tensor<5xi1>) -> tensor<4x!tt.ptr<f32>>
    // expected-error@+2 {{tta-to-memref: indirect_index is not mergeable}}
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%idx1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @indirect_on_block_ptr(%base: !tt.ptr<f16>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr = tta.make_addr %base to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], shape: [4, 4], order: [1, 0] : <f16> to !tt.ptr<tensor<4x4xf16>>
    %idx = "tta.reindex"(%addr, %offsets) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (!tt.ptr<tensor<4x4xf16>>, tensor<4xi32>) -> !tt.ptr<tensor<4x4xf16>>
    // expected-error@+2 {{tta-to-memref: indirect reindex on block pointer is unsupported}}
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x4xf16>>) -> tensor<4x4xf16>
    tt.return
  }
}

// -----

module {
  tt.func @unsupported_address_chain(%src: !tt.ptr<f32>) {
    %range = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %base = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    // expected-error@+2 {{tta-to-memref: unsupported address chain}}
    // expected-error@+1 {{failed to legalize operation 'tta.load' that was explicitly marked illegal}}
    %val = "tta.load"(%ptrs) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
  tt.func @atomic_tensor_unsupported(%ptr: !tt.ptr<i32>, %off: tensor<4xi32>, %val: tensor<4xi32>) {
    // expected-error@+2 {{tta-to-memref: tensor tta.atomic is unsupported}}
    // expected-error@+1 {{failed to legalize operation 'tta.atomic' that was explicitly marked illegal}}
    %r = "tta.atomic"(%ptr, %off, %val) <{kind = "add"}> : (!tt.ptr<i32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    tt.return
  }
}

// -----

module {
  tt.func @atomic_cas_tensor_unsupported(%ptr: !tt.ptr<i32>, %off: tensor<4xi32>, %cmp: tensor<4xi32>, %val: tensor<4xi32>) {
    // expected-error@+2 {{tta-to-memref: tensor tta.atomic_cas is unsupported}}
    // expected-error@+1 {{failed to legalize operation 'tta.atomic_cas' that was explicitly marked illegal}}
    %r = "tta.atomic_cas"(%ptr, %off, %cmp, %val) : (!tt.ptr<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    tt.return
  }
}
