// RUN: triton-xyz-opt --split-input-file --triton-to-tta-structured %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @snapshot_indirect_1d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) -> tensor<4xf32> {
// CHECK:           %[[IDX_ADDR:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[IDX:.*]] = "tta.load"(%[[IDX_ADDR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>) -> tensor<4xi32>
// CHECK:           %[[SRC_ADDR:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[SRC_IDX:.*]] = "tta.indirect_reindex"(%[[SRC_ADDR]], %[[IDX]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL:.*]] = "tta.load"(%[[SRC_IDX]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
// CHECK:           tt.return %[[VAL]] : tensor<4xf32>
// CHECK:         }
  tt.func @snapshot_indirect_1d(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>) -> tensor<4xf32> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %offsets = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_base, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %src_ptrs : tensor<4x!tt.ptr<f32>>
    tt.return %val : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @snapshot_indirect_broadcast_2d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) -> tensor<4x4xf32> {
// CHECK:           %[[IDX_ADDR:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[IDX:.*]] = "tta.load"(%[[IDX_ADDR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>) -> tensor<4xi32>
// CHECK:           %[[SRC_ADDR:.*]] = tta.make_addr %[[ARG0]] to sizes: [4, 4], strides: [1, 0], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[SRC_IDX:.*]] = "tta.indirect_reindex"(%[[SRC_ADDR]], %[[IDX]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL:.*]] = "tta.load"(%[[SRC_IDX]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<4x4xf32>
// CHECK:           tt.return %[[VAL]] : tensor<4x4xf32>
// CHECK:         }
  tt.func @snapshot_indirect_broadcast_2d(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>) -> tensor<4x4xf32> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %idx_val = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %idx_2d = tt.expand_dims %idx_val {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %idx_bcast = tt.broadcast %idx_2d : tensor<4x1xi32> -> tensor<4x4xi32>
    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_base, %idx_bcast : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %val = tt.load %src_ptrs : tensor<4x4x!tt.ptr<f32>>
    tt.return %val : tensor<4x4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @snapshot_indirect_structured_non_gather_dim(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) -> tensor<4x4xf32> {
// CHECK:           %[[IDX_ADDR:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[IDX:.*]] = "tta.load"(%[[IDX_ADDR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>) -> tensor<4xi32>
// CHECK:           %[[SRC_ADDR:.*]] = tta.make_addr %[[ARG0]] to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[SRC_IDX:.*]] = "tta.indirect_reindex"(%[[SRC_ADDR]], %[[IDX]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL:.*]] = "tta.load"(%[[SRC_IDX]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<4x4xf32>
// CHECK:           tt.return %[[VAL]] : tensor<4x4xf32>
// CHECK:         }
  tt.func @snapshot_indirect_structured_non_gather_dim(%src: !tt.ptr<f32>, %idx: !tt.ptr<i32>) -> tensor<4x4xf32> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %idx_base = tt.splat %idx : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %idx_ptrs = tt.addptr %idx_base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %idx_val = tt.load %idx_ptrs : tensor<4x!tt.ptr<i32>>
    %idx_row = tt.expand_dims %idx_val {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>

    %c4_i32 = arith.constant 4 : i32
    %stride = tt.splat %c4_i32 : i32 -> tensor<4x1xi32>
    %row_offsets = arith.muli %idx_row, %stride : tensor<4x1xi32>

    %col = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %row_bcast = tt.broadcast %row_offsets : tensor<4x1xi32> -> tensor<4x4xi32>
    %col_bcast = tt.broadcast %col : tensor<1x4xi32> -> tensor<4x4xi32>
    %offsets = arith.addi %row_bcast, %col_bcast : tensor<4x4xi32>

    %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_base, %offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %val = tt.load %src_ptrs : tensor<4x4x!tt.ptr<f32>>
    tt.return %val : tensor<4x4xf32>
  }
}
