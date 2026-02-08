// RUN: triton-xyz-opt --split-input-file --verify-diagnostics %s

module {
  tt.func @invalid_reindex_missing_indirect_dim(%arg0: !tt.ptr<f32>) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{indirect_dim is required when indirect_index is present}}
    %r = "tta.reindex"(%addr, %idx) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2xi32>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_indirect_dim_without_indirect(%arg0: !tt.ptr<f32>) {
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{indirect_dim requires indirect_index}}
    %r = "tta.reindex"(%addr) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_mask_without_indirect(%arg0: !tt.ptr<f32>) {
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{mask requires indirect_index}}
    %r = "tta.reindex"(%addr, %mask) <{operandSegmentSizes = array<i32: 1, 0, 0, 1>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2xi1>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_indirect_dim_oob(%arg0: !tt.ptr<f32>) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{indirect_dim is out of bounds}}
    %r = "tta.reindex"(%addr, %idx) <{indirect_dim = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2xi32>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_indirect_index_type(%arg0: !tt.ptr<f32>) {
    %idx = arith.constant dense<[[0, 1]]> : tensor<1x2xi32>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{indirect_index must be a 1D tensor of int or index type}}
    %r = "tta.reindex"(%addr, %idx) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<1x2xi32>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_mask_not_1d_i1(%arg0: !tt.ptr<f32>) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[[true, false]]> : tensor<1x2xi1>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{mask must be a 1D tensor of i1}}
    %r = "tta.reindex"(%addr, %idx, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2xi32>, tensor<1x2xi1>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_mask_size_mismatch(%arg0: !tt.ptr<f32>) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[true, false, true]> : tensor<3xi1>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{mask size must match indirect_index size}}
    %r = "tta.reindex"(%addr, %idx, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2xi32>, tensor<3xi1>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_offsets_rank_mismatch(%arg0: !tt.ptr<f32>) {
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{offsets must match address rank}}
    %r = "tta.reindex"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_reindex_result_type_mismatch(%arg0: !tt.ptr<f32>) {
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{result type must match address type}}
    %r = "tta.reindex"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>}> : (tensor<2x2x!tt.ptr<f32>>) -> !tt.ptr<f32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_advance_rank_mismatch(%arg0: !tt.ptr<f32>) {
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{deltas must match address rank}}
    %r = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_advance_result_type_mismatch(%arg0: !tt.ptr<f32>) {
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{result type must match address type}}
    %r = "tta.advance"(%addr) <{static_deltas = array<i64: 1, 0>}> : (tensor<2x2x!tt.ptr<f32>>) -> !tt.ptr<f32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_kind(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %val = arith.constant 1.000000e+00 : f32
    // expected-error@+1 {{unsupported atomic kind: bad}}
    %r = "tta.atomic"(%arg0, %arg1, %val) <{kind = "bad"}> : (!tt.ptr<f32>, i32, f32) -> f32
    tt.return
  }
}
