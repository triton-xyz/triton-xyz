// RUN: triton-xyz-opt --split-input-file --triton-to-tta-unstructured %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @local_failure_cat_does_not_block(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_2]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_3]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[CONSTANT_2]] : i32 -> tensor<2xi32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_3:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<2xi32>
// CHECK:           %[[SPLAT_4:.*]] = tt.splat %[[ARG1]] : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
// CHECK:           %[[MAKE_RANGE_1:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_1]], %[[MAKE_RANGE_1]] : tensor<2xi32>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_2]], %[[MAKE_RANGE_1]] : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
// CHECK:           %[[ADDI_2:.*]] = arith.addi %[[SPLAT_3]], %[[MAKE_RANGE_1]] : tensor<2xi32>
// CHECK:           %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_4]], %[[MAKE_RANGE_1]] : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
// CHECK:           %[[CAT_0:.*]] = tt.cat %[[ADDPTR_0]], %[[ADDPTR_1]] {tta.fallback, tta.fallback_reason = "multi_base_cat_unsupported"} : tensor<2x!tt.ptr<f32>> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[CAT_0]] : tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[LOAD_0]] : tensor<4xf32>
// CHECK:           %[[SPLAT_5:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_3:.*]] = arith.addi %[[SPLAT_5]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG2]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_1]], %[[ADDI_3]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[VAL_2]], %[[ADDF_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @local_failure_cat_does_not_block(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %r4 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

    %good_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %good_ptrs = tt.addptr %good_base, %r4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %good = tt.load %good_ptrs : tensor<4x!tt.ptr<f32>>

    %b0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %b1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %r2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %c0 = tt.addptr %b0, %r2 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %c1 = tt.addptr %b1, %r2 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %cat = tt.cat %c0, %c1 : tensor<2x!tt.ptr<f32>> -> tensor<4x!tt.ptr<f32>>
    %bad = tt.load %cat : tensor<4x!tt.ptr<f32>>

    %sum = arith.addf %good, %bad : tensor<4xf32>

    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %r4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_other_not_scalar(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_1]] : i32 -> tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[ARG2]] : i32 -> tensor<4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[MAKE_RANGE_0]], %[[SPLAT_2]] : tensor<4xi32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[SPLAT_3:.*]] = tt.splat %[[CONSTANT_2]] : f32 -> tensor<4xf32>
// CHECK:           %[[SPLAT_4:.*]] = tt.splat %[[CONSTANT_3]] : f32 -> tensor<4xf32>
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[SPLAT_3]], %[[SPLAT_4]] : tensor<4xf32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]], %[[CMPI_0]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]], %[[CMPI_0]], %[[ADDF_0]] {tta.fallback, tta.fallback_reason = "other_not_scalar_splat"} : tensor<4x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_5:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_5]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_1]], %[[ADDI_1]], %[[CMPI_0]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[VAL_1]], %[[LOAD_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @fallback_other_not_scalar(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>

    %limit = tt.splat %arg2 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %s0 = tt.splat %c0 : f32 -> tensor<4xf32>
    %s1 = tt.splat %c1 : f32 -> tensor<4xf32>
    %other = arith.addf %s0, %s1 : tensor<4xf32>

    %v = tt.load %in_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>

    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %v, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_addptr_in_if_reason(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_2]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_5]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG3]] -> (!tt.ptr<f32>) {
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[ARG0]], %[[CONSTANT_3]] {tta.fallback, tta.fallback_reason = "addptr_in_scf_if_unsupported"} : !tt.ptr<f32>, i32
// CHECK:             scf.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } else {
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[ARG1]], %[[CONSTANT_4]] {tta.fallback, tta.fallback_reason = "addptr_in_scf_if_unsupported"} : !tt.ptr<f32>, i32
// CHECK:             scf.yield %[[ADDPTR_1]] : !tt.ptr<f32>
// CHECK:           }
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[IF_0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDPTR_2:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_2]] : tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[LOAD_0]] : tensor<4xf32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SPLAT_2]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG2]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_2:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_1]], %[[ADDI_1]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           "tta.store"(%[[VAL_2]], %[[ADDF_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @fallback_addptr_in_if_reason(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i1) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>

    %good_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %good_ptrs = tt.addptr %good_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %good = tt.load %good_ptrs : tensor<4x!tt.ptr<f32>>

    %sel_ptr = scf.if %arg3 -> (!tt.ptr<f32>) {
      %a = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %a : !tt.ptr<f32>
    } else {
      %b = tt.addptr %arg1, %c2_i32 : !tt.ptr<f32>, i32
      scf.yield %b : !tt.ptr<f32>
    }
    %bad_base = tt.splat %sel_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %bad_ptrs = tt.addptr %bad_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %bad = tt.load %bad_ptrs : tensor<4x!tt.ptr<f32>>

    %sum = arith.addf %good, %bad : tensor<4xf32>
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_atomic_kind_unsupported(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[ARG0]], %[[CONSTANT_0]] : !tt.ptr<i32>, i32
// CHECK:           %[[ATOMIC_RMW_0:.*]] = tt.atomic_rmw umax, acq_rel, gpu, %[[ADDPTR_0]], %[[ARG1]], %[[ARG2]] {tta.fallback, tta.fallback_reason = "atomic_kind_unsupported"} : (!tt.ptr<i32>, i32, i1) -> i32
// CHECK:           tt.return
// CHECK:         }
  tt.func @fallback_atomic_kind_unsupported(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) {
    %r = tt.atomic_rmw umax, acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @overwrite_existing_fallback_reason(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[ARG0]], %[[CONSTANT_0]] : !tt.ptr<i32>, i32
// CHECK:           %[[ATOMIC_RMW_0:.*]] = tt.atomic_rmw umax, acq_rel, gpu, %[[ADDPTR_0]], %[[ARG1]], %[[ARG2]] {tta.fallback, tta.fallback_reason = "atomic_kind_unsupported"} : (!tt.ptr<i32>, i32, i1) -> i32
// CHECK:           tt.return
// CHECK:         }
  tt.func @overwrite_existing_fallback_reason(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) {
    %r = tt.atomic_rmw umax, acq_rel, gpu, %arg0, %arg1, %arg2 {tta.fallback, tta.fallback_reason = "pre_marked_reason"} : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_ptr_to_int_tensor_result(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> (tensor<4xf32>, tensor<4xi64>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[CONSTANT_0]] : i32 -> tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4xi32>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[PTR_TO_INT_0:.*]] = tt.ptr_to_int %[[ADDPTR_0]] {tta.fallback, tta.fallback_reason = "ptr_to_int_tensor_result_unsupported"} : tensor<4x!tt.ptr<f32>> -> tensor<4xi64>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = "tta.load"(%[[VAL_0]], %[[CONSTANT_1]]) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
// CHECK:           tt.return %[[VAL_1]], %[[PTR_TO_INT_0]] : tensor<4xf32>, tensor<4xi64>
// CHECK:         }
  tt.func @fallback_ptr_to_int_tensor_result(%arg0: !tt.ptr<f32>) -> (tensor<4xf32>, tensor<4xi64>) {
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ints = tt.ptr_to_int %ptrs : tensor<4x!tt.ptr<f32>> -> tensor<4xi64>
    %vals = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
    tt.return %vals, %ints : tensor<4xf32>, tensor<4xi64>
  }
}
