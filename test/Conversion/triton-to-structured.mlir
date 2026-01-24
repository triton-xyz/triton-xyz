// RUN: triton-shared-opt --split-input-file --triton-to-structured %s | FileCheck %s
// RUN: triton-shared-opt --split-input-file --triton-to-structured="run-prepass-only=true" %s | FileCheck %s --check-prefix=PREPASS

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
// PREPASS-LABEL:   tt.func @basic_addptr_1d(
// PREPASS-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// PREPASS-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// PREPASS:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// PREPASS:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// PREPASS:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// PREPASS:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]] : tensor<4x!tt.ptr<f32>>
// PREPASS:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// PREPASS:           %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// PREPASS:           tt.store %[[ADDPTR_1]], %[[LOAD_0]] : tensor<4x!tt.ptr<f32>>
// PREPASS:           tt.return
// PREPASS:         }
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
// PREPASS-LABEL:   tt.func @masked_1d(
// PREPASS-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// PREPASS-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// PREPASS-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// PREPASS:           %[[CONSTANT_0:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// PREPASS:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// PREPASS:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// PREPASS:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// PREPASS:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// PREPASS:           %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_0]] : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
// PREPASS:           %[[SPLAT_2:.*]] = tt.splat %[[ARG2]] : i32 -> tensor<8xi32>
// PREPASS:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[MAKE_RANGE_0]], %[[SPLAT_2]] : tensor<8xi32>
// PREPASS:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]], %[[CMPI_0]], %[[CONSTANT_0]] : tensor<8x!tt.ptr<f32>>
// PREPASS:           tt.store %[[ADDPTR_1]], %[[LOAD_0]], %[[CMPI_0]] : tensor<8x!tt.ptr<f32>>
// PREPASS:           tt.return
// PREPASS:         }
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
// CHECK-LABEL:   tt.func @loop_ptr_iterargs(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_3]], %[[VAL_2:.*]] = %[[INDEX_CAST_0]]) -> (tensor<4xf32>, index) {
// CHECK:             %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [4], strides: [1], offsets: {{\[}}%[[VAL_2]]], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:             %[[VAL_3:.*]] = "tts.load"(%[[MAKE_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[VAL_3]] : tensor<4xf32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[VAL_0]] : index
// CHECK:             scf.yield %[[ADDF_0]], %[[ADDI_0]] : tensor<4xf32>, index
// CHECK:           }
// CHECK:           %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           "tts.store"(%[[MAKE_TPTR_1]], %[[VAL_4:.*]]#0) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
// PREPASS-LABEL:   tt.func @loop_ptr_iterargs(
// PREPASS-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// PREPASS-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// PREPASS:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// PREPASS:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// PREPASS:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// PREPASS:           %[[CONSTANT_3:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// PREPASS:           %[[ADDPTR_0:.*]] = tt.addptr %[[ARG0]], %[[ARG1]] : !tt.ptr<f32>, i32
// PREPASS:           %[[VAL_0:.*]], %[[VAL_1:.*]] = "tts.get_structured_state"(%[[ADDPTR_0]]) <{resultSegmentSizes = array<i32: 1, 1, 0>}> : (!tt.ptr<f32>) -> (!tt.ptr<f32>, index)
// PREPASS:           %[[FOR_0:.*]]:3 = scf.for %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] iter_args(%[[VAL_3:.*]] = %[[CONSTANT_3]], %[[VAL_4:.*]] = %[[VAL_0]], %[[VAL_5:.*]] = %[[VAL_1]]) -> (tensor<4xf32>, !tt.ptr<f32>, index) {
// PREPASS:             %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// PREPASS:             %[[SPLAT_0:.*]] = tt.splat %[[VAL_4]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// PREPASS:             %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// PREPASS:             %[[LOAD_0:.*]] = tt.load %[[ADDPTR_1]] : tensor<4x!tt.ptr<f32>>
// PREPASS:             %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[LOAD_0]] : tensor<4xf32>
// PREPASS:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_2]] : index to i32
// PREPASS:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_4]], %[[INDEX_CAST_0]] : !tt.ptr<f32>, i32
// PREPASS:             %[[VAL_6:.*]], %[[VAL_7:.*]] = "tts.get_structured_state"(%[[ADDPTR_2]]) <{resultSegmentSizes = array<i32: 1, 1, 0>}> : (!tt.ptr<f32>) -> (!tt.ptr<f32>, index)
// PREPASS:             scf.yield %[[ADDF_0]], %[[VAL_6]], %[[VAL_7]] : tensor<4xf32>, !tt.ptr<f32>, index
// PREPASS:           }
// PREPASS:           %[[MAKE_RANGE_1:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// PREPASS:           %[[SPLAT_1:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// PREPASS:           %[[ADDPTR_3:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_1]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// PREPASS:           tt.store %[[ADDPTR_3]], %[[VAL_8:.*]]#0 : tensor<4x!tt.ptr<f32>>
// PREPASS:           tt.return
// PREPASS:         }
  tt.func @loop_ptr_iterargs(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %ptr0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
    %sum, %ptr = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init, %p = %ptr0) -> (tensor<4xf32>, !tt.ptr<f32>) {
      %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %base = tt.splat %p : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %ptrs = tt.addptr %base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %val = tt.load %ptrs : tensor<4x!tt.ptr<f32>>
      %next = arith.addf %acc, %val : tensor<4xf32>
      %i_i32 = arith.index_cast %i : index to i32
      %p_next = tt.addptr %p, %i_i32 : !tt.ptr<f32>, i32
      scf.yield %next, %p_next : tensor<4xf32>, !tt.ptr<f32>
    }
    %out_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %out_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %out_range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %sum : tensor<4x!tt.ptr<f32>>
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
// PREPASS-LABEL:   tt.func @block_ptr_basic(
// PREPASS-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// PREPASS:           %[[CONSTANT_0:.*]] = arith.constant 4 : i64
// PREPASS:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
// PREPASS:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// PREPASS:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// PREPASS:           %[[MAKE_TENSOR_PTR_0:.*]] = tt.make_tensor_ptr %[[ARG0]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]], {{\[}}%[[CONSTANT_0]], %[[CONSTANT_1]]], {{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] {order = array<i32: 1, 0>} : <tensor<4x4xf16>>
// PREPASS:           %[[ADVANCE_0:.*]] = tt.advance %[[MAKE_TENSOR_PTR_0]], {{\[}}%[[CONSTANT_2]], %[[CONSTANT_3]]] : <tensor<4x4xf16>>
// PREPASS:           %[[LOAD_0:.*]] = tt.load %[[ADVANCE_0]] : !tt.ptr<tensor<4x4xf16>>
// PREPASS:           tt.store %[[MAKE_TENSOR_PTR_0]], %[[LOAD_0]] : !tt.ptr<tensor<4x4xf16>>
// PREPASS:           tt.return
// PREPASS:         }
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
// PREPASS-LABEL:   tt.func @gather_scatter_2d(
// PREPASS-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// PREPASS-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// PREPASS-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// PREPASS:           %[[CONSTANT_0:.*]] = arith.constant dense<4> : tensor<4x1xi32>
// PREPASS:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// PREPASS:           %[[SPLAT_0:.*]] = tt.splat %[[ARG1]] : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
// PREPASS:           %[[ADDPTR_0:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
// PREPASS:           %[[LOAD_0:.*]] = tt.load %[[ADDPTR_0]] : tensor<4x!tt.ptr<i32>>
// PREPASS:           %[[EXPAND_DIMS_0:.*]] = tt.expand_dims %[[LOAD_0]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// PREPASS:           %[[MULI_0:.*]] = arith.muli %[[EXPAND_DIMS_0]], %[[CONSTANT_0]] : tensor<4x1xi32>
// PREPASS:           %[[EXPAND_DIMS_1:.*]] = tt.expand_dims %[[MAKE_RANGE_0]] {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
// PREPASS:           %[[BROADCAST_0:.*]] = tt.broadcast %[[MULI_0]] : tensor<4x1xi32> -> tensor<4x4xi32>
// PREPASS:           %[[BROADCAST_1:.*]] = tt.broadcast %[[EXPAND_DIMS_1]] : tensor<1x4xi32> -> tensor<4x4xi32>
// PREPASS:           %[[ADDI_0:.*]] = arith.addi %[[BROADCAST_0]], %[[BROADCAST_1]] : tensor<4x4xi32>
// PREPASS:           %[[SPLAT_1:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
// PREPASS:           %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_1]], %[[ADDI_0]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// PREPASS:           %[[LOAD_1:.*]] = tt.load %[[ADDPTR_1]] : tensor<4x4x!tt.ptr<f32>>
// PREPASS:           %[[EXPAND_DIMS_2:.*]] = tt.expand_dims %[[MAKE_RANGE_0]] {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
// PREPASS:           %[[MULI_1:.*]] = arith.muli %[[EXPAND_DIMS_2]], %[[CONSTANT_0]] : tensor<4x1xi32>
// PREPASS:           %[[BROADCAST_2:.*]] = tt.broadcast %[[MULI_1]] : tensor<4x1xi32> -> tensor<4x4xi32>
// PREPASS:           %[[ADDI_1:.*]] = arith.addi %[[BROADCAST_2]], %[[BROADCAST_1]] : tensor<4x4xi32>
// PREPASS:           %[[SPLAT_2:.*]] = tt.splat %[[ARG2]] : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
// PREPASS:           %[[ADDPTR_2:.*]] = tt.addptr %[[SPLAT_2]], %[[ADDI_1]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
// PREPASS:           tt.store %[[ADDPTR_2]], %[[LOAD_1]] : tensor<4x4x!tt.ptr<f32>>
// PREPASS:           tt.return
// PREPASS:         }
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
