// RUN: triton-xyz-opt --split-input-file --triton-to-structured --triton-to-unstructured --triton-to-tta-structured --triton-to-tta-unstructured --remove-dead-values --canonicalize %s | FileCheck %s

module {
// CHECK-LABEL: tt.func @basic_addptr_1d(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: %[[V:.*]] = "tta.load"(%[[A0]])
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: "tta.store"(%[[A1]], %[[V]])
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
// CHECK-LABEL: tt.func @masked_1d(
// CHECK: %[[IN:.*]] = tta.make_addr %arg0 to sizes: [8]
// CHECK: %[[OUT:.*]] = tta.make_addr %arg1 to sizes: [8]
// CHECK: "tta.load"(%[[IN]], %{{.*}}, %{{.*}})
// CHECK: "tta.store"(%[[OUT]], %{{.*}}, %{{.*}})
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
// CHECK-LABEL: tt.func @block_ptr_basic(
// CHECK: tta.make_addr %arg0 to sizes: [4, 4]
// CHECK: order: [1, 0]
// CHECK: "tta.load"(%
// CHECK: "tta.store"(%
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
// CHECK-LABEL: tt.func @gather_scatter_2d(
// CHECK: %[[IDX_ADDR:.*]] = tta.make_addr %arg1 to sizes: [4]
// CHECK: %[[IDX:.*]] = "tta.load"(%[[IDX_ADDR]])
// CHECK: %[[SRC_ADDR:.*]] = tta.make_addr %arg0 to sizes: [4, 4]
// CHECK: %[[R:.*]] = "tta.reindex"(%[[SRC_ADDR]], %[[IDX]])
// CHECK: %[[V:.*]] = "tta.load"(%[[R]])
// CHECK: %[[DST_ADDR:.*]] = tta.make_addr %arg2 to sizes: [4, 4]
// CHECK: "tta.store"(%[[DST_ADDR]], %[[V]])
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
