// RUN: triton-xyz-opt --split-input-file --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @basic_addptr_1d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[MAKE_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
// CHECK:           %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[MAKE_TPTR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @basic_addptr_1d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %in_ptrs : tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %val : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @masked_1d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 8 : index
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
// CHECK:           %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG1]] to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[MINSI_0:.*]] = arith.minsi %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:           %[[MAXSI_0:.*]] = arith.maxsi %[[MINSI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[MAKE_TPTR_0]], %[[MAXSI_0]], %[[CONSTANT_0]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<8x!tt.ptr<f32>>, index, f32) -> tensor<8xf32>
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[MINSI_1:.*]] = arith.minsi %[[INDEX_CAST_1]], %[[CONSTANT_2]] : index
// CHECK:           %[[MAXSI_1:.*]] = arith.maxsi %[[MINSI_1]], %[[CONSTANT_1]] : index
// CHECK:           "tts.store"(%[[MAKE_TPTR_1]], %[[VAL_0]], %[[MAXSI_1]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @masked_1d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<8xf32>
    %limit = tt.splat %arg2 : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<8xi32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<8x!tt.ptr<f32>>
    tt.store %out_ptrs, %val, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_ptr_basic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [4, 4], strides: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_2]]], offsets: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]], shape: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], order: [1, 0] : <f16> to !tt.ptr<tensor<4x4xf16>>
// CHECK:           %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG0]] to sizes: [4, 4], strides: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_2]]], offsets: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_2]]], shape: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], order: [1, 0] : <f16> to !tt.ptr<tensor<4x4xf16>>
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[MAKE_TPTR_1]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x4xf16>>) -> tensor<4x4xf16>
// CHECK:           "tts.store"(%[[MAKE_TPTR_0]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x4xf16>>, tensor<4x4xf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_ptr_basic(%arg0: !tt.ptr<f16>) {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
    %adv = tt.advance %ptr, [%c0_i32, %c1_i32] : <tensor<4x4xf16>>
    %val = tt.load %adv : !tt.ptr<tensor<4x4xf16>>
    tt.store %ptr, %val : !tt.ptr<tensor<4x4xf16>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @gather_scatter_2d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<4x!tt.ptr<i32>>
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[MAKE_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<i32>>) -> tensor<4xi32>
// CHECK:           %[[MAKE_GATHER_SCATTER_TPTR_0:.*]] = tts.make_gather_scatter_tptr %[[ARG0]] to sizes: [4, 4] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_0]], strides: {{\[}}%[[CONSTANT_0]], 1], offsets: [0, 0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4x4xf32>>
// CHECK:           %[[VAL_1:.*]] = "tts.load"(%[[MAKE_GATHER_SCATTER_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<4x4xf32>>) -> tensor<4x4xf32>
// CHECK:           %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG2]] to sizes: [4, 4], strides: {{\[}}%[[CONSTANT_0]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<4x4x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[MAKE_TPTR_1]], %[[VAL_1]]) <{static_mask_dims = array<i64>}> : (tensor<4x4x!tt.ptr<f32>>, tensor<4x4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @gather_scatter_2d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %idx = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %idx_row = tt.expand_dims %idx {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %c4_i32 = arith.constant 4 : i32
    %stride = tt.splat %c4_i32 : i32 -> tensor<4x1xi32>
    %row_offsets = arith.muli %idx_row, %stride : tensor<4x1xi32>
    %col = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_bcast = tt.broadcast %row_offsets : tensor<4x1xi32> -> tensor<4x4xi32>
    %col_bcast = tt.broadcast %col : tensor<1x4xi32> -> tensor<4x4xi32>
    %offsets = arith.addi %row_bcast, %col_bcast : tensor<4x4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %val = tt.load %ptrs : tensor<4x4x!tt.ptr<f32>>
    %row = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %row_stride = tt.splat %c4_i32 : i32 -> tensor<4x1xi32>
    %row_linear = arith.muli %row, %row_stride : tensor<4x1xi32>
    %row_linear_bcast = tt.broadcast %row_linear : tensor<4x1xi32> -> tensor<4x4xi32>
    %linear_offsets = arith.addi %row_linear_bcast, %col_bcast : tensor<4x4xi32>
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %linear_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    tt.store %out_ptrs, %val : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @scalar_addptr_splat(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [4], strides: [1], offsets: {{\[}}%[[INDEX_CAST_0]]], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[MAKE_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
// CHECK:           %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[MAKE_TPTR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @scalar_addptr_splat(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %base = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %in_ptrs : tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %val : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @row_major_2d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 4], strides: {{\[}}%[[CONSTANT_0]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x4x!tt.ptr<f32>>
// CHECK:           %[[VAL_0:.*]] = "tts.load"(%[[MAKE_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x4x!tt.ptr<f32>>) -> tensor<2x4xf32>
// CHECK:           %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 4], strides: {{\[}}%[[CONSTANT_0]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<2x4x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[MAKE_TPTR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (tensor<2x4x!tt.ptr<f32>>, tensor<2x4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @row_major_2d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %row_exp = tt.expand_dims %row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %col_exp = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_bcast = tt.broadcast %row_exp : tensor<2x1xi32> -> tensor<2x4xi32>
    %col_bcast = tt.broadcast %col_exp : tensor<1x4xi32> -> tensor<2x4xi32>
    %c4 = arith.constant 4 : i32
    %stride = tt.splat %c4 : i32 -> tensor<2x4xi32>
    %row_linear = arith.muli %row_bcast, %stride : tensor<2x4xi32>
    %offsets = arith.addi %row_linear, %col_bcast : tensor<2x4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %val = tt.load %in_ptrs : tensor<2x4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %out_ptrs, %val : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}
