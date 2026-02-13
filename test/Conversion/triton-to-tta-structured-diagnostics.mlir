// RUN: triton-xyz-opt --split-input-file --triton-to-tta-structured %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @fallback_mask_rank_not_1d(
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
  tt.func @fallback_mask_rank_not_1d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
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
// CHECK-LABEL:   tt.func @fallback_mask_analysis_failed(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[SPLAT_2:.*]] = tt.splat %[[ARG2]] : i32 -> tensor<4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[MAKE_RANGE_0]], %[[SPLAT_2]] : tensor<4xi32>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]], %[[CMPI_0]], %[[CONSTANT_0]] {tta.fallback, tta.fallback_reason = "mask_analysis_failed"} : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.store %[[ADDPTR_1]], %[[LOAD_0]], %[[CMPI_0]] {tta.fallback, tta.fallback_reason = "mask_analysis_failed"} : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
  tt.func @fallback_mask_analysis_failed(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg2 : i32 -> tensor<4xi32>
    %mask = arith.cmpi eq, %range, %limit : tensor<4xi32>
    %zero = arith.constant 0.0 : f32
    %other = tt.splat %zero : f32 -> tensor<4xf32>
    %val = tt.load %in_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.store %out_ptrs, %val, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_other_not_scalar_splat(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : i32 -> tensor<4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[MAKE_RANGE_0]], %[[SPLAT_1]] : tensor<4xi32>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]], %[[CMPI_0]], %[[CONSTANT_0]] {tta.fallback, tta.fallback_reason = "other_not_scalar_splat"} : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.return %[[LOAD_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @fallback_other_not_scalar_splat(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<4xf32> {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %limit = tt.splat %arg1 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<4xi32>
    %other = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
    %val = tt.load %ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return %val : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_boundary_check_not_supported(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]] {boundaryCheck = array<i32: 0>, padding = 2 : i32, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"} : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.store %[[ADDPTR_0]], %[[LOAD_0]] {boundaryCheck = array<i32: 0>, tta.fallback, tta.fallback_reason = "boundary_check_not_supported"} : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
  tt.func @fallback_boundary_check_not_supported(%arg0: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %ptrs {boundaryCheck = array<i32: 0>, padding = 2 : i32} : tensor<4x!tt.ptr<f32>>
    tt.store %ptrs, %val {boundaryCheck = array<i32: 0>} : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @preserve_premarked_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]] {tta.fallback, tta.fallback_reason = "pre_marked_load"} : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.store %[[ADDPTR_1]], %[[LOAD_0]] {tta.fallback, tta.fallback_reason = "pre_marked_store"} : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
  tt.func @preserve_premarked_load_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %in_ptrs {tta.fallback, tta.fallback_reason = "pre_marked_load"} : tensor<4x!tt.ptr<f32>>
    tt.store %out_ptrs, %val {tta.fallback, tta.fallback_reason = "pre_marked_store"} : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_tensor_ptr_unhandled(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_TENSOR_PTR_0:.*]] = tt.make_tensor_ptr %[[ARG0]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_1]]], {{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "tensor_ptr_unhandled"} : <tensor<4x4xf16>>
// CHECK:           tt.return %[[MAKE_TENSOR_PTR_0]] : !tt.ptr<tensor<4x4xf16>>
// CHECK:         }
  tt.func @fallback_tensor_ptr_unhandled(%arg0: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
    tt.return %ptr : !tt.ptr<tensor<4x4xf16>>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @preserve_premarked_tensor_ptr_reason(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_TENSOR_PTR_0:.*]] = tt.make_tensor_ptr %[[ARG0]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_1]]], {{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "pre_marked_reason"} : <tensor<4x4xf16>>
// CHECK:           tt.return %[[MAKE_TENSOR_PTR_0]] : !tt.ptr<tensor<4x4xf16>>
// CHECK:         }
  tt.func @preserve_premarked_tensor_ptr_reason(%arg0: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "pre_marked_reason"} : <tensor<4x4xf16>>
    tt.return %ptr : !tt.ptr<tensor<4x4xf16>>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @override_premarked_tensor_ptr_reason(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[MAKE_TENSOR_PTR_0:.*]] = tt.make_tensor_ptr %[[ARG0]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_1]]], {{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] {order = array<i32: 1, 0>, tta.fallback, tta.fallback_reason = "tensor_ptr_unhandled"} : <tensor<4x4xf16>>
// CHECK:           tt.return %[[MAKE_TENSOR_PTR_0]] : !tt.ptr<tensor<4x4xf16>>
// CHECK:         }
  tt.func @override_premarked_tensor_ptr_reason(%arg0: !tt.ptr<f16>) -> !tt.ptr<tensor<4x4xf16>> {
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %ptr = tt.make_tensor_ptr %arg0, [%c4_i64, %c4_i64], [%c4_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>, tta.fallback_reason = "preexisting_reason"} : <tensor<4x4xf16>>
    tt.return %ptr : !tt.ptr<tensor<4x4xf16>>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @fallback_gather_scatter_tts_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
// CHECK:           %[[MAKE_GATHER_SCATTER_TPTR_0:.*]] = tts.make_gather_scatter_tptr %[[ARG0]] to sizes: [4, 4] gather_scatter_dim: 0 gather_scatter_offset: %[[CONSTANT_0]], strides: [4, 1], offsets: [0, 0] : tensor<4xi32>  <f32> to !tt.ptr<tensor<4x4xf32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[MAKE_GATHER_SCATTER_TPTR_0]] {tta.fallback, tta.fallback_reason = "ptr_expr_analysis_failed"} : !tt.ptr<tensor<4x4xf32>>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG1]] to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           "tta.store"(%[[MAKE_ADDR_0]], %[[LOAD_0]]) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<4x4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @fallback_gather_scatter_tts_ptr(%base: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %src = tts.make_gather_scatter_tptr %base to sizes: [4, 4] gather_scatter_dim: 0 gather_scatter_offset: %offsets, strides: [4, 1], offsets: [0, 0] : tensor<4xi32> <f32> to !tt.ptr<tensor<4x4xf32>>
    %dst_tptr = tts.make_tptr %dst to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tt.ptr<tensor<4x4xf32>>
    %val = tt.load %src : !tt.ptr<tensor<4x4xf32>>
    tt.store %dst_tptr, %val : !tt.ptr<tensor<4x4xf32>>
    tt.return
  }
}
