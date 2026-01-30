// RUN: triton-xyz-opt --split-input-file --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s
// RUN: triton-xyz-opt --split-input-file --triton-to-structured="run-prepass-only=true" --remove-dead-values --canonicalize %s | FileCheck %s --check-prefix=PREPASS

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
// PREPASS:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_2:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] iter_args(%[[VAL_3:.*]] = %[[CONSTANT_3]], %[[VAL_4:.*]] = %[[VAL_0]]) -> (tensor<4xf32>, !tt.ptr<f32>) {
// PREPASS:             %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// PREPASS:             %[[SPLAT_0:.*]] = tt.splat %[[VAL_4]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// PREPASS:             %[[ADDPTR_1:.*]] = tt.addptr %[[SPLAT_0]], %[[MAKE_RANGE_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// PREPASS:             %[[LOAD_0:.*]] = tt.load %[[ADDPTR_1]] : tensor<4x!tt.ptr<f32>>
// PREPASS:             %[[ADDF_0:.*]] = arith.addf %[[VAL_3]], %[[LOAD_0]] : tensor<4xf32>
// PREPASS:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_2]] : index to i32
// PREPASS:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_4]], %[[INDEX_CAST_0]] : !tt.ptr<f32>, i32
// PREPASS:             %[[VAL_5:.*]], %[[VAL_6:.*]] = "tts.get_structured_state"(%[[ADDPTR_2]]) <{resultSegmentSizes = array<i32: 1, 1, 0>}> : (!tt.ptr<f32>) -> (!tt.ptr<f32>, index)
// PREPASS:             scf.yield %[[ADDF_0]], %[[VAL_5]] : tensor<4xf32>, !tt.ptr<f32>
// PREPASS:           }
// PREPASS:           %[[MAKE_RANGE_1:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// PREPASS:           %[[SPLAT_1:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// PREPASS:           %[[ADDPTR_3:.*]] = tt.addptr %[[SPLAT_1]], %[[MAKE_RANGE_1]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// PREPASS:           tt.store %[[ADDPTR_3]], %[[VAL_7:.*]]#0 : tensor<4x!tt.ptr<f32>>
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
// PREPASS-LABEL:   tt.func @loop_i64_iterargs(
// PREPASS:           "tts.get_structured_state"(%{{.*}}) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<4xi64>) -> (tensor<4xi64>, index, index)
  tt.func @loop_i64_iterargs(%arg0: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %range_i64 = arith.extsi %range : tensor<4xi32> to tensor<4xi64>
    %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %range_i64) -> (tensor<4xi64>) {
      %next = arith.addi %acc, %range_i64 : tensor<4xi64>
      scf.yield %next : tensor<4xi64>
    }
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptrs = tt.addptr %base, %sum : tensor<4x!tt.ptr<i32>>, tensor<4xi64>
    %val = tt.load %ptrs : tensor<4x!tt.ptr<i32>>
    tt.store %ptrs, %val : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
