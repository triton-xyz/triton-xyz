// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta %s | FileCheck %s

// TODO: rm this test

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @loop_ptr_iter_args_lowering(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[ARG2]] step %[[CONSTANT_3]] iter_args(%[[VAL_2:.*]] = %[[GENERIC_0]], %[[VAL_3:.*]] = %[[GENERIC_0]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             linalg.fill ins(%[[CONSTANT_5]] : f32) outs(%[[ALLOC_0]] : memref<4xf32>)
// CHECK:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[VAL_2]]{{\[}}%[[VAL_4]]] : tensor<4xi32>
// CHECK:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_4]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             scf.for %[[VAL_5:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[VAL_3]]{{\[}}%[[VAL_5]]] : tensor<4xi32>
// CHECK:               %[[INDEX_CAST_2:.*]] = arith.index_cast %[[EXTRACT_1]] : i32 to index
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: {{\[}}%[[INDEX_CAST_2]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_5]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:             %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_2]], %[[GENERIC_0]] : tensor<4xi32>, tensor<4xi32>) outs(%[[VAL_2]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:               linalg.yield %[[ADDI_0]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_3]], %[[GENERIC_0]] : tensor<4xi32>, tensor<4xi32>) outs(%[[VAL_3]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32):
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:               linalg.yield %[[ADDI_1]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             scf.yield %[[GENERIC_1]], %[[GENERIC_2]] : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func public @loop_ptr_iter_args_lowering(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>

    %res:2 = scf.for %iv = %c0_i32 to %arg2 step %c1_i32 iter_args(%in = %in_ptrs, %out = %out_ptrs) -> (tensor<4x!tt.ptr<f32>>, tensor<4x!tt.ptr<f32>>) : i32 {
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
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @if_ptr_merge_lowering(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[ARG2]], %[[ARG0]], %[[ARG1]] : memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_3]] : f32) outs(%[[ALLOC_0]] : memref<4xf32>)
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[GENERIC_0]]{{\[}}%[[VAL_1]]] : tensor<4xi32>
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[SELECT_0]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_1]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
  tt.func public @if_ptr_merge_lowering(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %pred: i1) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %in0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %in1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr0 = tt.addptr %in0, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %ptr1 = tt.addptr %in1, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>

    %chosen = scf.if %pred -> (tensor<4x!tt.ptr<f32>>) {
      scf.yield %ptr0 : tensor<4x!tt.ptr<f32>>
    } else {
      scf.yield %ptr1 : tensor<4x!tt.ptr<f32>>
    }

    %v = tt.load %chosen : tensor<4x!tt.ptr<f32>>
    %out = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %out_ptrs, %v : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}
