// RUN: triton-xyz-opt --split-input-file --triton-to-tta-unstructured %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func public @masked_gather_scatter(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[ARG2]] : i32 -> tensor<4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[MAKE_RANGE_0]], %[[SPLAT_2]] : tensor<4xi32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[SPLAT_3:.*]] = tt.splat %[[CONSTANT_2]] : f32 -> tensor<4xf32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]], %[[CMPI_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_2]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.reindex"(%[[MAKE_ADDR_1]], %[[ADDI_1]], %[[CMPI_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[VAL_2]], %[[VAL_1]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg2 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.store %out_ptrs, %val, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @bitcast_ptr_chain(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> tensor<4xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[BITCAST_0:.*]] = tt.bitcast %[[ARG0]] : !tt.ptr<f32> -> !tt.ptr<i32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[BITCAST_0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<i32, 1, 1>, tensor<4xi32>) -> !tta.addr<i32, 1, 1>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_1]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>, i32) -> tensor<4xi32>
// CHECK:           tt.return %[[VAL_1]] : tensor<4xi32>
// CHECK:         }
  tt.func @bitcast_ptr_chain(%arg0: !tt.ptr<f32>) -> tensor<4xi32> {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %bc = tt.bitcast %arg0 : !tt.ptr<f32> -> !tt.ptr<i32>
    %base = tt.splat %bc : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptrs = tt.addptr %base, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<i32>>
    tt.return %vals : tensor<4xi32>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @int_to_ptr_root_lowering(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> tensor<4xf32> {
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[INT_TO_PTR_0:.*]] = tt.int_to_ptr %[[ARG0]] : i64 -> !tt.ptr<f32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[INT_TO_PTR_0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_1]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:           tt.return %[[VAL_1]] : tensor<4xf32>
// CHECK:         }
  tt.func @int_to_ptr_root_lowering(%arg0: i64) -> tensor<4xf32> {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %p = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
    tt.return %vals : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @scalar_atomic_rmw_to_tta(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.atomic"(%[[MAKE_ADDR_0]], %[[CONSTANT_0]], %[[ARG1]], %[[ARG2]]) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
// CHECK:           tt.return %[[VAL_0]] : i32
// CHECK:         }
  tt.func @scalar_atomic_rmw_to_tta(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) -> i32 {
    %a0 = tt.atomic_rmw add, acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return %a0 : i32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @offset_width_upgrade(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 5 : i64
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_2]] : i64 -> tensor<4xi64>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[EXTSI_0:.*]] = arith.extsi %[[SPLAT_1]] : tensor<4xi32> to tensor<4xi64>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[EXTSI_0]], %[[SPLAT_0]] : tensor<4xi64>
// CHECK:           %[[EXTSI_1:.*]] = arith.extsi %[[MAKE_RANGE_0]] : tensor<4xi32> to tensor<4xi64>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[EXTSI_1]] : tensor<4xi64>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[ADDI_1]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi64>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_3]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[SPLAT_2]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.reindex"(%[[MAKE_ADDR_1]], %[[ADDI_2]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[VAL_2]], %[[VAL_1]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func public @offset_width_upgrade(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c5_i64 = arith.constant 5 : i64
    %off64 = tt.splat %c5_i64 : i64 -> tensor<4xi64>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr0 = tt.addptr %base, %off64 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %ptr1 = tt.addptr %ptr0, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %ptr1 : tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %val : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @loop_ptr_iter_args(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[ARG2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[ADDI_0]], %[[VAL_2:.*]] = %[[ADDI_1]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK:             %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_3:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[VAL_1]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_4:.*]] = "tta.load"(%[[VAL_3]], %[[CONSTANT_4]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:             %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_5:.*]] = "tta.reindex"(%[[MAKE_ADDR_1]], %[[VAL_2]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:             "tta.store"(%[[VAL_5]], %[[VAL_4]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_2]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:             scf.yield %[[ADDI_2]], %[[ADDI_3]] : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @loop_ptr_iter_args(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %res:2 = scf.for %iv = %c0_i32 to %arg2 step %c1_i32 iter_args(%in = %in_ptrs, %out = %out_ptrs) -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>)  : i32 {
      %val = tt.load %in : tensor<4x!tt.ptr<f32>>
      tt.store %out, %val : tensor<4x!tt.ptr<f32>>
      %next_in = tt.addptr %in, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %next_out = tt.addptr %out, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next_in, %next_out : tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @masked_2d_fallback(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[MAKE_RANGE_1:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[EXPAND_DIMS_0:.*]] = tt.expand_dims %[[MAKE_RANGE_0]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK:           %[[EXPAND_DIMS_1:.*]] = tt.expand_dims %[[MAKE_RANGE_1]] {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
// CHECK:           %[[BROADCAST_0:.*]] = tt.broadcast %[[EXPAND_DIMS_0]] : tensor<2x1xi32> -> tensor<2x4xi32>
// CHECK:           %[[BROADCAST_1:.*]] = tt.broadcast %[[EXPAND_DIMS_1]] : tensor<1x4xi32> -> tensor<2x4xi32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : i32
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_2]] : i32 -> tensor<2x4xi32>
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[BROADCAST_0]], %[[SPLAT_0]] : tensor<2x4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[BROADCAST_1]] : tensor<2x4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<2x4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_1]], %[[ADDI_0]] : tensor<2x4xi32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[ARG2]] : i32 -> tensor<2x4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[ADDI_0]], %[[SPLAT_2]] : tensor<2x4xi32>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[SPLAT_3:.*]] = tt.splat %[[CONSTANT_3]] : f32 -> tensor<2x4xf32>
// CHECK:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[ADDI_1]] {{\[\[}}0, 1]] : tensor<2x4xi32> into tensor<8xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[COLLAPSE_SHAPE_1:.*]] = tensor.collapse_shape %[[CMPI_0]] {{\[\[}}0, 1]] : tensor<2x4xi1> into tensor<8xi1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[COLLAPSE_SHAPE_0]], %[[COLLAPSE_SHAPE_1]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<8xi32>, tensor<8xi1>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_3]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<8xf32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[VAL_1]] {{\[\[}}0, 1]] output_shape [2, 4] : tensor<8xf32> into tensor<2x4xf32>
// CHECK:           %[[SPLAT_4:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<2x4xi32>
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[SPLAT_4]], %[[ADDI_0]] : tensor<2x4xi32>
// CHECK:           %[[COLLAPSE_SHAPE_2:.*]] = tensor.collapse_shape %[[ADDI_2]] {{\[\[}}0, 1]] : tensor<2x4xi32> into tensor<8xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[COLLAPSE_SHAPE_3:.*]] = tensor.collapse_shape %[[CMPI_0]] {{\[\[}}0, 1]] : tensor<2x4xi1> into tensor<8xi1>
// CHECK:           %[[VAL_2:.*]] = "tta.reindex"(%[[MAKE_ADDR_1]], %[[COLLAPSE_SHAPE_2]], %[[COLLAPSE_SHAPE_3]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<8xi32>, tensor<8xi1>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[COLLAPSE_SHAPE_4:.*]] = tensor.collapse_shape %[[EXPAND_SHAPE_0]] {{\[\[}}0, 1]] : tensor<2x4xf32> into tensor<8xf32>
// CHECK:           "tta.store"(%[[VAL_2]], %[[COLLAPSE_SHAPE_4]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<8xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func public @masked_2d_fallback(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
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
// CHECK-LABEL:   tt.func @scalar_atomic_rmw_and_cas_to_tta(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.atomic"(%[[MAKE_ADDR_0]], %[[CONSTANT_0]], %[[ARG1]], %[[ARG3]]) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG0]] to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.atomic"(%[[MAKE_ADDR_1]], %[[CONSTANT_0]], %[[ARG2]]) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
// CHECK:           %[[MAKE_ADDR_2:.*]] = tta.make_addr %[[ARG0]] to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.atomic_cas"(%[[MAKE_ADDR_2]], %[[CONSTANT_0]], %[[ARG1]], %[[ARG2]]) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[VAL_2]] : i32
// CHECK:           tt.return %[[ADDI_1]] : i32
// CHECK:         }
  tt.func @scalar_atomic_rmw_and_cas_to_tta(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32, %arg3: i1) -> i32 {
    %a0 = tt.atomic_rmw add, acq_rel, gpu, %arg0, %arg1, %arg3 : (!tt.ptr<i32>, i32, i1) -> i32
    %a1 = tt.atomic_rmw exch, acq_rel, gpu, %arg0, %arg2 : (!tt.ptr<i32>, i32) -> i32
    %a2 = tt.atomic_cas acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i32) -> i32
    %sum0 = arith.addi %a0, %a1 : i32
    %sum = arith.addi %sum0, %a2 : i32
    tt.return %sum : i32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @ptr_to_int_scalar_materialized(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i64 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_2]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.reindex"(%[[MAKE_ADDR_1]], %[[ADDI_1]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[VAL_2]], %[[VAL_1]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[CONSTANT_1]], %[[ARG2]] : i32
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[ARG0]], %[[ADDI_2]] : !tt.ptr<f32>, i32
// CHECK:           %[[PTR_TO_INT_0:.*]] = tt.ptr_to_int %[[ADDPTR_0]] : !tt.ptr<f32> -> i64
// CHECK:           tt.return %[[PTR_TO_INT_0]] : i64
// CHECK:         }
  tt.func @ptr_to_int_scalar_materialized(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) -> i64 {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
    %outbase = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %outptrs = tt.addptr %outbase, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %outptrs, %vals : tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
    %i = tt.ptr_to_int %p : !tt.ptr<f32> -> i64
    tt.return %i : i64
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @make_tensor_ptr_accumulate_offset(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f16> to !tta.addr<f16, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f16, 1, 1>, tensor<4xi32>) -> !tta.addr<f16, 1, 1>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_2]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 1, 1>, f16) -> tensor<4xf16>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : i64
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[CONSTANT_1]], %[[ARG2]] : i32
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[CONSTANT_5]] : i32
// CHECK:           %[[MAKE_TENSOR_PTR_0:.*]] = tt.make_tensor_ptr %[[ARG0]], {{\[}}%[[CONSTANT_3]], %[[CONSTANT_3]]], {{\[}}%[[CONSTANT_3]], %[[CONSTANT_4]]], {{\[}}%[[ADDI_2]], %[[CONSTANT_5]]] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[MAKE_TENSOR_PTR_0]] : !tt.ptr<tensor<4x4xf16>>
// CHECK:           tt.store %[[MAKE_TENSOR_PTR_0]], %[[LOAD_0]] : !tt.ptr<tensor<4x4xf16>>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_3:.*]] = arith.addi %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f16> to !tta.addr<f16, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.reindex"(%[[MAKE_ADDR_1]], %[[ADDI_3]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f16, 1, 1>, tensor<4xi32>) -> !tta.addr<f16, 1, 1>
// CHECK:           "tta.store"(%[[VAL_2]], %[[VAL_1]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 1, 1>, tensor<4xf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @make_tensor_ptr_accumulate_offset(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32) {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

    %seed = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %seed_ptrs = tt.addptr %seed, %r : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    %seed_vals = tt.load %seed_ptrs : tensor<4x!tt.ptr<f16>>

    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %base2 = tt.addptr %arg0, %arg2 : !tt.ptr<f16>, i32
    %tptr = tt.make_tensor_ptr %base2, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
    %loaded = tt.load %tptr : !tt.ptr<tensor<4x4xf16>>
    tt.store %tptr, %loaded : !tt.ptr<tensor<4x4xf16>>

    %outbase = tt.splat %arg1 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %outptrs = tt.addptr %outbase, %r : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    tt.store %outptrs, %seed_vals : tensor<4x!tt.ptr<f16>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @select_same_base_ptr_offsets_to_tta(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) -> tensor<4xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<1> : tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[CONSTANT_1]] : tensor<4xi32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant dense<2> : tensor<4xi32>
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[MAKE_RANGE_0]], %[[CONSTANT_2]] : tensor<4xi32>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant dense<0> : tensor<4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[REMSI_0]], %[[CONSTANT_3]] : tensor<4xi32>
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_0]], %[[ADDI_1]] : tensor<4xi1>, tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to !tta.addr<i32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[SELECT_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<i32, 1, 1>, tensor<4xi32>) -> !tta.addr<i32, 1, 1>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_4]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<i32, 1, 1>, i32) -> tensor<4xi32>
// CHECK:           tt.return %[[VAL_1]] : tensor<4xi32>
// CHECK:         }
  tt.func @select_same_base_ptr_offsets_to_tta(%arg0: !tt.ptr<i32>) -> tensor<4xi32> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptr0 = tt.addptr %base, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %c1 = arith.constant dense<1> : tensor<4xi32>
    %ptr1 = tt.addptr %ptr0, %c1 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %c2 = arith.constant dense<2> : tensor<4xi32>
    %rem = arith.remsi %range, %c2 : tensor<4xi32>
    %c0 = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi eq, %rem, %c0 : tensor<4xi32>
    %ptrs = arith.select %mask, %ptr0, %ptr1 : tensor<4xi1>, tensor<4x!tt.ptr<i32>>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<i32>>
    tt.return %vals : tensor<4xi32>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @scalar_ptr_store_fallback_to_tta(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[ARG1]] : i32
// CHECK:           %[[FROM_ELEMENTS_0:.*]] = tensor.from_elements %[[ADDI_0]] : tensor<1xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [1], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[FROM_ELEMENTS_0]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: 0>}> : (!tta.addr<f32, 1, 1>, tensor<1xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[FROM_ELEMENTS_1:.*]] = tensor.from_elements %[[ARG2]] : tensor<1xf32>
// CHECK:           "tta.store"(%[[VAL_0]], %[[FROM_ELEMENTS_1]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<1xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @scalar_ptr_store_fallback_to_tta(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: f32) {
    %ptr = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
    tt.store %ptr, %arg2 : !tt.ptr<f32>
    tt.return
  }
}
