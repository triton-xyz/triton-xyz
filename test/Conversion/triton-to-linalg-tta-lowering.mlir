// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta %s | FileCheck %s

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @vector_add(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) {
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1]>> to memref<4xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[ALLOC_1:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_1]], %[[ALLOC_1]] : memref<4xf32, strided<[1]>> to memref<4xf32>
// CHECK:           %[[TO_TENSOR_1:.*]] = bufferization.to_tensor %[[ALLOC_1]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[TO_TENSOR_0]], %[[TO_TENSOR_1]] : tensor<4xf32>, tensor<4xf32>) outs(%[[TO_TENSOR_0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32):
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:             linalg.yield %[[ADDF_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ARG2]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[GENERIC_0]] in writable %[[REINTERPRET_CAST_2]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
  tt.func @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %lhs_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %rhs_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %out_base = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %rhs_ptrs = tt.addptr %rhs_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %out_ptrs = tt.addptr %out_base, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %lhs = tt.load %lhs_ptrs : tensor<4x!tt.ptr<f32>>
    %rhs = tt.load %rhs_ptrs : tensor<4x!tt.ptr<f32>>
    %sum = arith.addf %lhs, %rhs : tensor<4xf32>
    tt.store %out_ptrs, %sum : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   func.func @gather_scatter_2d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xi32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [4], strides: [1] : memref<*xi32> to memref<4xi32, strided<[1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xi32, strided<[1]>> to memref<4xi32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xi32> to tensor<4xi32>
// CHECK:           %[[ALLOC_1:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[MULI_0]]], sizes: [1, 4], strides: [4, 1] : memref<*xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_1]]{{\[}}%[[VAL_0]], 0] [1, 4] [1, 1] : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_1]], %[[SUBVIEW_0]] : memref<1x4xf32, strided<[4, 1], offset: ?>> to memref<1x4xf32, strided<[4, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_1:.*]] = bufferization.to_tensor %[[ALLOC_1]] restrict writable : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:           %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ARG2]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<*xf32> to memref<4x4xf32, strided<[4, 1]>>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_2]] : memref<4x4xf32, strided<[4, 1]>> to memref<4x4xf32, strided<[?, 1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_1]] in writable %[[CAST_0]] : (tensor<4x4xf32>, memref<4x4xf32, strided<[?, 1]>>) -> ()
// CHECK:           return
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
// CHECK-LABEL:   func.func @atomic_scalar_tta_route(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[CONSTANT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[ARG1]] : i32
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[ARG2]], %[[ADDI_0]], %[[VAL_0]] : i32
// CHECK:             memref.atomic_yield %[[SELECT_0]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func @atomic_scalar_tta_route(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i1) {
    %r = tt.atomic_rmw add, acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i1) -> i32
    %u = arith.addi %r, %arg1 : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   func.func @atomic_cas_scalar_tta_route(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[CONSTANT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_0]], %[[ARG1]] : i32
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ARG2]], %[[VAL_0]] : i32
// CHECK:             memref.atomic_yield %[[SELECT_0]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func @atomic_cas_scalar_tta_route(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32) {
    %r = tt.atomic_cas acq_rel, gpu, %arg0, %arg1, %arg2 : (!tt.ptr<i32>, i32, i32) -> i32
    %u = arith.addi %r, %arg2 : i32
    tt.return
  }
}
