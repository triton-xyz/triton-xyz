// RUN: triton-xyz-opt --split-input-file --triton-to-tta-structured %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @basic_addptr_1d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
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
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [8], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[MINSI_0:.*]] = arith.minsi %[[INDEX_CAST_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[MAXSI_0:.*]] = arith.maxsi %[[MINSI_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]], %[[MAXSI_0]], %[[CONSTANT_2]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (!tta.addr<f32, 1, 1>, index, f32) -> tensor<8xf32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [8], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[MINSI_1:.*]] = arith.minsi %[[INDEX_CAST_1]], %[[CONSTANT_1]] : index
// CHECK:           %[[MAXSI_1:.*]] = arith.maxsi %[[MINSI_1]], %[[CONSTANT_0]] : index
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[VAL_0]], %[[MAXSI_1]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (!tta.addr<f32, 1, 1>, tensor<8xf32>, index) -> ()
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
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4, 4], strides: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_2]]], offsets: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_2]]], layout: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG0]] to sizes: [4, 4], strides: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_2]]], offsets: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]], layout: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>, tensor<4x4xf16>) -> ()
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
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[IDX:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>) -> tensor<4xi32>
// CHECK:           %[[LOAD_0:.*]] = tt.load {{.*}} {tta.fallback, tta.fallback_reason = "indirect non-gather dim must be singleton or broadcast"} : tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG2]] to sizes: [4, 4], strides: {{\[}}%[[CONSTANT_0]], 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[LOAD_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<4x4xf32>) -> ()
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
// CHECK-LABEL:   tt.func @gather_1d_indirect(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[IDX:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>) -> tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[SRC_IDX:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_1]], %[[IDX]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL:.*]] = "tta.load"(%[[SRC_IDX]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
// CHECK:           %[[MAKE_ADDR_2:.*]] = tta.make_addr %[[ARG2]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_2]], %[[VAL]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK-NOT:       tta.fallback
// CHECK:           tt.return
// CHECK:         }
  tt.func @gather_1d_indirect(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %offsets = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_base, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %src_ptrs : tensor<4x!tt.ptr<f32>>
    %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %dst_ptrs, %val : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @gather_2d_singleton_indirect(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[IDX_ADDR:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[IDX:.*]] = "tta.load"(%[[IDX_ADDR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>) -> tensor<4xi32>
// CHECK:           %[[SRC_ADDR:.*]] = tta.make_addr %[[ARG0]] to sizes: [4, 1], strides: {{\[}}1, {{[01]}}], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[SRC_IDX:.*]] = "tta.indirect_reindex"(%[[SRC_ADDR]], %[[IDX]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL:.*]] = "tta.load"(%[[SRC_IDX]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<4x1xf32>
// CHECK:           %[[DST_ADDR:.*]] = tta.make_addr %[[ARG2]] to sizes: [4, 1], strides: {{\[}}1, {{[01]}}], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           "tta.store"(%[[DST_ADDR]], %[[VAL]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<4x1xf32>) -> ()
// CHECK-NOT:       tta.fallback
// CHECK:           tt.return
// CHECK:         }
  tt.func @gather_2d_singleton_indirect(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %offsets = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %offsets2d = tt.expand_dims %offsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>

    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_base, %offsets2d : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    %val = tt.load %src_ptrs : tensor<4x1x!tt.ptr<f32>>

    %range2d = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x1x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_base, %range2d : tensor<4x1x!tt.ptr<f32>>, tensor<4x1xi32>
    tt.store %dst_ptrs, %val : tensor<4x1x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @row_major_2d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 4], strides: {{\[}}%[[CONSTANT_0]], 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [2, 4], strides: {{\[}}%[[CONSTANT_0]], 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
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

// -----

module {
// CHECK-LABEL:   tt.func @gather_2d_broadcast_indirect(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[IDX_ADDR:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[IDX:.*]] = "tta.load"(%[[IDX_ADDR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>) -> tensor<4xi32>
// CHECK:           %[[SRC_ADDR:.*]] = tta.make_addr %[[ARG0]] to sizes: [4, 4], strides: [1, 0], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[SRC_IDX:.*]] = "tta.indirect_reindex"(%[[SRC_ADDR]], %[[IDX]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL:.*]] = "tta.load"(%[[SRC_IDX]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<4x4xf32>
// CHECK:           %[[DST_ADDR:.*]] = tta.make_addr %[[ARG2]] to sizes: [4, 4], strides: {{\[}}%[[CONSTANT_0:.*]], 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           "tta.store"(%[[DST_ADDR]], %[[VAL]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<4x4xf32>) -> ()
// CHECK-NOT:       tta.fallback
// CHECK:           tt.return
// CHECK:         }
  tt.func @gather_2d_broadcast_indirect(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %offsets = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %offsets2d = tt.expand_dims %offsets {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %offsets4 = tt.broadcast %offsets2d : tensor<4x1xi32> -> tensor<4x4xi32>

    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_base, %offsets4 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %val = tt.load %src_ptrs : tensor<4x4x!tt.ptr<f32>>

    %row = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %col = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_bcast = tt.broadcast %row : tensor<4x1xi32> -> tensor<4x4xi32>
    %col_bcast = tt.broadcast %col : tensor<1x4xi32> -> tensor<4x4xi32>
    %c4 = arith.constant 4 : i32
    %stride = tt.splat %c4 : i32 -> tensor<4x4xi32>
    %row_linear = arith.muli %row_bcast, %stride : tensor<4x4xi32>
    %linear_offsets = arith.addi %row_linear, %col_bcast : tensor<4x4xi32>
    %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_base, %linear_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    tt.store %dst_ptrs, %val : tensor<4x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @prebuilt_tta_addr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @prebuilt_tta_addr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %a0 = tta.make_addr %arg0 to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %v = "tta.load"(%a0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
    %a1 = tta.make_addr %arg1 to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%a1, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @masked_2d_fallback(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<4> : tensor<2x4xi32>
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[MAKE_RANGE_1:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[EXPAND_DIMS_0:.*]] = tt.expand_dims %[[MAKE_RANGE_0]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK:           %[[EXPAND_DIMS_1:.*]] = tt.expand_dims %[[MAKE_RANGE_1]] {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
// CHECK:           %[[BROADCAST_0:.*]] = tt.broadcast %[[EXPAND_DIMS_0]] : tensor<2x1xi32> -> tensor<2x4xi32>
// CHECK:           %[[BROADCAST_1:.*]] = tt.broadcast %[[EXPAND_DIMS_1]] : tensor<1x4xi32> -> tensor<2x4xi32>
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[BROADCAST_0]], %[[CONSTANT_1]] : tensor<2x4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[BROADCAST_1]] : tensor<2x4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[ADDI_0]] : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG2]] : i32 -> tensor<2x4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[ADDI_0]], %[[SPLAT_1]] : tensor<2x4xi32>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]], %[[CMPI_0]], %[[CONSTANT_0]] {tta.fallback, tta.fallback_reason = "mask_rank_not_1d"} : tensor<2x4x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[ARG1]] : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
// CHECK:           %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_2]], %[[ADDI_0]] : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
// CHECK:           tt.store %[[ADDPTR_1]], %[[LOAD_0]], %[[CMPI_0]] {tta.fallback, tta.fallback_reason = "mask_rank_not_1d"} : tensor<2x4x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
  tt.func @masked_2d_fallback(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
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
    %limit = tt.splat %arg2 : i32 -> tensor<2x4xi32>
    %mask = arith.cmpi slt, %offsets, %limit : tensor<2x4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<2x4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<2x4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %out_ptrs, %val, %mask : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @from_tts_make_tptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK-NOT:       tt.load
// CHECK-NOT:       tt.store
// CHECK:           tt.return
// CHECK:         }
  tt.func @from_tts_make_tptr(%base: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src = tts.make_tptr %base to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %dst_tptr = tts.make_tptr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %val = tt.load %src : tensor<4x!tt.ptr<f32>>
    tt.store %dst_tptr, %val : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @advance_from_tts_make_tptr_block(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 8 : index
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [8, 4], strides: [4, 1], offsets: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_0]]], layout: [8, 4] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.load"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<8x4xf32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG0]] to sizes: [8, 4], strides: [4, 1], offsets: {{\[}}%[[CONSTANT_1]], %[[CONSTANT_0]]], layout: [8, 4] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_1]], %[[VAL_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<8x4xf32>) -> ()
// CHECK-NOT:       tt.advance
// CHECK-NOT:       tta.fallback
// CHECK:           tt.return
// CHECK:         }
  tt.func @advance_from_tts_make_tptr_block(%base: !tt.ptr<f32>) {
    %c2 = arith.constant 2 : i32
    %c1 = arith.constant 1 : i32
    %src = tts.make_tptr %base to sizes: [8, 4], strides: [4, 1], offsets: [0, 0], shape: [8, 4], order: [1, 0] : <f32> to !tt.ptr<tensor<8x4xf32>>
    %adv = tt.advance %src, [%c2, %c1] : <tensor<8x4xf32>>
    %val = tt.load %adv : !tt.ptr<tensor<8x4xf32>>
    tt.store %adv, %val : !tt.ptr<tensor<8x4xf32>>
    tt.return
  }
}
