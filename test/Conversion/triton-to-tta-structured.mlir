// RUN: triton-xyz-opt --split-input-file --triton-to-tta-structured --remove-dead-values --canonicalize %s | FileCheck %s
// CHECK-NOT: tts.

module {
// CHECK-LABEL: tt.func @basic_addptr_1d(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: %[[V:.*]] = "tta.load"(%[[A0]]
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
// CHECK: %[[V:.*]] = "tta.load"(%[[IN]], %{{.*}}, %{{.*}})
// CHECK: %[[OUT:.*]] = tta.make_addr %arg1 to sizes: [8]
// CHECK: "tta.store"(%[[OUT]], %[[V]], %{{.*}})
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
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4, 4]
// CHECK: order: [1, 0]
// CHECK: %[[VAL:.*]] = "tta.load"(%[[A0]])
// CHECK: %[[A1:.*]] = tta.make_addr %arg0 to sizes: [4, 4]
// CHECK: order: [1, 0]
// CHECK: "tta.store"(%[[A1]], %[[VAL]])
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
// CHECK: %[[IDX:.*]] = "tta.load"(%[[IDX_ADDR]]
// CHECK: tt.load %{{.*}} : tensor<4x4x!tt.ptr<f32>>
// CHECK: "tta.store"(%{{.*}}, %{{.*}})
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
// CHECK-LABEL: tt.func @row_major_2d(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [2, 4], strides: [%{{.*}}, 1], offsets: [0, 0], shape: [0, 0], order: []
// CHECK: %[[V:.*]] = "tta.load"(%[[A0]])
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [2, 4], strides: [%{{.*}}, 1], offsets: [0, 0], shape: [0, 0], order: []
// CHECK: "tta.store"(%[[A1]], %[[V]])
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
// CHECK-LABEL: tt.func @prebuilt_tta_addr(
// CHECK: %[[A0:.*]] = tta.make_addr %arg0 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: %[[V:.*]] = "tta.load"(%[[A0]])
// CHECK: %[[A1:.*]] = tta.make_addr %arg1 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: "tta.store"(%[[A1]], %[[V]])
  tt.func @prebuilt_tta_addr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %a0 = tta.make_addr %arg0 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    %v = "tta.load"(%a0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
    %a1 = tta.make_addr %arg1 to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%a1, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL: tt.func @masked_2d_fallback(
// CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4x!tt.ptr<f32>>
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
