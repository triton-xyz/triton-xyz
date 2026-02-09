// RUN: triton-xyz-opt --split-input-file --verify-diagnostics %s

module {
  tt.func @invalid_make_addr_strides_rank_mismatch(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{strides must match sizes rank}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_make_addr_offsets_rank_mismatch(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{offsets must match sizes rank}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_make_addr_shape_rank_mismatch(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{shape must match sizes rank}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_make_addr_order_non_empty_for_tensor_ptr(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{order must be empty for tensor-of-ptr result}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [1, 0] : <f32> to tensor<2x2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_make_addr_order_rank_mismatch_for_block_ptr(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{order length must match sizes rank for block pointer result}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [2, 2], order: [1] : <f32> to !tt.ptr<tensor<2x2xf32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_make_addr_order_not_permutation_for_block_ptr(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{order must be a permutation of [0, rank)}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [2, 2], order: [0, 0] : <f32> to !tt.ptr<tensor<2x2xf32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_make_addr_tensor_shape_mismatch(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{tensor result shape must match sizes}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x3x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_make_addr_block_shape_mismatch(%arg0: !tt.ptr<f32>) {
    // expected-error@+1 {{pointer pointee shape must match sizes}}
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [2, 2], order: [1, 0] : <f32> to !tt.ptr<tensor<2x3xf32>>
    tt.return
  }
}

// -----

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
  tt.func @invalid_load_mask_rank_mismatch(%arg0: !tt.ptr<f32>) {
    %m = arith.constant 1 : index
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{mask_dims must be empty or match pointer rank}}
    %v = "tta.load"(%addr, %m) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<2x2x!tt.ptr<f32>>, index) -> tensor<2x2xf32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_load_other_type_mismatch(%arg0: !tt.ptr<f32>) {
    %other = arith.constant 1 : i32
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{other type must match pointer pointee element type}}
    %v = "tta.load"(%addr, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, i32) -> tensor<2x2xf32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_load_result_shape_mismatch(%arg0: !tt.ptr<f32>) {
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{result shape must match loaded pointer shape}}
    %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x3xf32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_store_mask_rank_mismatch(%arg0: !tt.ptr<f32>) {
    %m = arith.constant 1 : index
    %val = arith.constant dense<0.0> : tensor<2x2xf32>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{mask_dims must be empty or match pointer rank}}
    "tta.store"(%addr, %val, %m) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, index) -> ()
    tt.return
  }
}

// -----

module {
  tt.func @invalid_store_value_shape_mismatch(%arg0: !tt.ptr<f32>) {
    %val = arith.constant dense<0.0> : tensor<2x3xf32>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{value shape must match stored pointer shape}}
    "tta.store"(%addr, %val) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x3xf32>) -> ()
    tt.return
  }
}

// -----

module {
  tt.func @invalid_store_value_element_type_mismatch(%arg0: !tt.ptr<f32>) {
    %val = arith.constant dense<0> : tensor<2x2xi32>
    %addr = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    // expected-error@+1 {{value element type must match pointer pointee type}}
    "tta.store"(%addr, %val) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>) -> ()
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

// -----

module {
  tt.func @invalid_atomic_offset_value_rank_mismatch(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %val = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
    // expected-error@+1 {{offset and value must both be scalars or both be tensors}}
    %r = "tta.atomic"(%arg0, %arg1, %val) <{kind = "add"}> : (!tt.ptr<f32>, i32, tensor<2xf32>) -> tensor<2xf32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_value_type_mismatch(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %val = arith.constant 1 : i32
    // expected-error@+1 {{value element type must match pointer pointee type}}
    %r = "tta.atomic"(%arg0, %arg1, %val) <{kind = "add"}> : (!tt.ptr<f32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_fadd_requires_float(%arg0: !tt.ptr<i32>, %arg1: i32) {
    %val = arith.constant 1 : i32
    // expected-error@+1 {{fadd requires floating-point value type}}
    %r = "tta.atomic"(%arg0, %arg1, %val) <{kind = "fadd"}> : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_bitwise_requires_integer(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %val = arith.constant 1.000000e+00 : f32
    // expected-error@+1 {{and requires integer value type}}
    %r = "tta.atomic"(%arg0, %arg1, %val) <{kind = "and"}> : (!tt.ptr<f32>, i32, f32) -> f32
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_cmpxchg_deprecated(%arg0: !tt.ptr<i32>, %arg1: i32) {
    %val = arith.constant 1 : i32
    // expected-error@+1 {{unsupported atomic kind: cmpxchg}}
    %r = "tta.atomic"(%arg0, %arg1, %val) <{kind = "cmpxchg"}> : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_cas_compare_value_type_mismatch(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32, %arg3: i64) {
    // expected-error@+1 {{compare type matches value type}}
    %r = "tta.atomic_cas"(%arg0, %arg1, %arg2, %arg3) : (!tt.ptr<i32>, i32, i32, i64) -> i64
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_cas_offset_value_rank_mismatch(%arg0: !tt.ptr<i32>, %arg1: i32) {
    %cmp = arith.constant dense<[1, 2]> : tensor<2xi32>
    %val = arith.constant dense<[3, 4]> : tensor<2xi32>
    // expected-error@+1 {{offset and value must both be scalars or both be tensors}}
    %r = "tta.atomic_cas"(%arg0, %arg1, %cmp, %val) : (!tt.ptr<i32>, i32, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_cas_tensor_shape_mismatch(%arg0: !tt.ptr<i32>) {
    %off = arith.constant dense<[0, 1]> : tensor<2xi32>
    %cmp = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %val = arith.constant dense<[4, 5, 6]> : tensor<3xi32>
    // expected-error@+1 {{offset and value tensor shapes must match}}
    %r = "tta.atomic_cas"(%arg0, %off, %cmp, %val) : (!tt.ptr<i32>, tensor<2xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_cas_value_type_mismatch(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: f32, %arg3: f32) {
    // expected-error@+1 {{value element type must match pointer pointee type}}
    %r = "tta.atomic_cas"(%arg0, %arg1, %arg2, %arg3) : (!tt.ptr<i32>, i32, f32, f32) -> f32
    tt.return
  }
}

// -----

module {
  tt.func @invalid_atomic_cas_non_numeric(%arg0: !tt.ptr<!tt.ptr<i32>>, %arg1: i32, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) {
    // expected-error@+1 {{atomic_cas requires integer or floating-point value type}}
    %r = "tta.atomic_cas"(%arg0, %arg1, %arg2, %arg3) : (!tt.ptr<!tt.ptr<i32>>, i32, !tt.ptr<i32>, !tt.ptr<i32>) -> !tt.ptr<i32>
    tt.return
  }
}
