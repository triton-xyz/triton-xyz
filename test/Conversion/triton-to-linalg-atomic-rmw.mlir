// RUN: triton-xyz-opt --split-input-file --triton-to-linalg %s | FileCheck %s

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @atomic_rmw_tensor_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 4 : index
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_2:.*]] = %[[FILL_0]]) -> (tensor<4xi32>)  : i32 {
// CHECK:             %[[FILL_1:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:             %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_1]] : tensor<4xi32>, tensor<4xi32>) outs(%[[GENERIC_0]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_3]], %[[VAL_4]] : i32
// CHECK:               linalg.yield %[[ADDI_0]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             %[[FOR_1:.*]] = scf.for %[[VAL_6:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_5]] step %[[CONSTANT_4]] iter_args(%[[VAL_7:.*]] = %[[EMPTY_0]]) -> (tensor<4xi32>) {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[VAL_2]]{{\[}}%[[VAL_6]]] : tensor<4xi32>
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[GENERIC_1]]{{\[}}%[[VAL_6]]] : tensor<4xi32>
// CHECK:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_1]] : i32 to index
// CHECK:               %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xi32> to memref<?xi32>
// CHECK:               %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_1]]] : memref<?xi32> {
// CHECK:               ^bb0(%[[VAL_8:.*]]: i32):
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[VAL_8]], %[[EXTRACT_0]] : i32
// CHECK:                 memref.atomic_yield %[[ADDI_1]] : i32
// CHECK:               }
// CHECK:               %[[INSERT_0:.*]] = tensor.insert %[[GENERIC_ATOMIC_RMW_0]] into %[[VAL_7]]{{\[}}%[[VAL_6]]] : tensor<4xi32>
// CHECK:               scf.yield %[[INSERT_0]] : tensor<4xi32>
// CHECK:             }
// CHECK:             %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_2]], %[[FOR_1]] : tensor<4xi32>, tensor<4xi32>) outs(%[[VAL_2]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32):
// CHECK:               %[[ADDI_2:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:               linalg.yield %[[ADDI_2]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             scf.yield %[[GENERIC_2]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func public @atomic_rmw_tensor_ptr(%arg0: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<true> : tensor<4xi1>
    %c4_i32 = arith.constant 4 : i32
    %init = arith.constant dense<0> : tensor<4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %ptrs = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptrs2 = tt.addptr %ptrs, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %loop = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<4xi32>)  : i32 {
      %i_splat = tt.splat %i : i32 -> tensor<4xi32>
      %ptrs3 = tt.addptr %ptrs2, %i_splat : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %val = tt.atomic_rmw add, acq_rel, gpu, %ptrs3, %acc, %cst : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
      %sum = arith.addi %acc, %val : tensor<4xi32>
      scf.yield %sum : tensor<4xi32>
    }
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @atomic_cas_tensor_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 2 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]]  : i32 {
// CHECK:             %[[FILL_0:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:             %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_0]] : tensor<4xi32>, tensor<4xi32>) outs(%[[GENERIC_0]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32):
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:               linalg.yield %[[ADDI_0]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             scf.for %[[VAL_5:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_5]] step %[[CONSTANT_4]] {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[GENERIC_1]]{{\[}}%[[VAL_5]]] : tensor<4xi32>
// CHECK:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:               %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xi32> to memref<?xi32>
// CHECK:               %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_1]]] : memref<?xi32> {
// CHECK:               ^bb0(%[[VAL_6:.*]]: i32):
// CHECK:                 %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_6]], %[[CONSTANT_1]] : i32
// CHECK:                 %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONSTANT_6]], %[[VAL_6]] : i32
// CHECK:                 memref.atomic_yield %[[SELECT_0]] : i32
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func public @atomic_cas_tensor_ptr(%arg0: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %init = arith.constant dense<0> : tensor<4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %ptrs = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptrs2 = tt.addptr %ptrs, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %cmp = arith.constant dense<1> : tensor<4xi32>
    %val = arith.constant dense<2> : tensor<4xi32>
    %loop = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<4xi32>)  : i32 {
      %i_splat = tt.splat %i : i32 -> tensor<4xi32>
      %ptrs3 = tt.addptr %ptrs2, %i_splat : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %old = tt.atomic_cas acq_rel, gpu, %ptrs3, %cmp, %val : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
      %sum = arith.addi %acc, %old : tensor<4xi32>
      scf.yield %sum : tensor<4xi32>
    }
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @atomic_rmw_tensor_ptr_masked(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 4 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4xi1>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_0]] : tensor<4xi32>, tensor<4xi32>) outs(%[[EMPTY_1]] : tensor<4xi1>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1):
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[CMPI_0]] : i1
// CHECK:           } -> tensor<4xi1>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_4:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_6]] step %[[CONSTANT_5]] iter_args(%[[VAL_5:.*]] = %[[FILL_1]]) -> (tensor<4xi32>)  : i32 {
// CHECK:             %[[FILL_2:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:             %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_2]] : tensor<4xi32>, tensor<4xi32>) outs(%[[GENERIC_0]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:               linalg.yield %[[ADDI_0]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             %[[FOR_1:.*]] = scf.for %[[VAL_9:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_4]] step %[[CONSTANT_3]] iter_args(%[[VAL_10:.*]] = %[[EMPTY_0]]) -> (tensor<4xi32>) {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[VAL_5]]{{\[}}%[[VAL_9]]] : tensor<4xi32>
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[GENERIC_1]]{{\[}}%[[VAL_9]]] : tensor<4xi1>
// CHECK:               %[[EXTRACT_2:.*]] = tensor.extract %[[GENERIC_2]]{{\[}}%[[VAL_9]]] : tensor<4xi32>
// CHECK:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_2]] : i32 to index
// CHECK:               %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xi32> to memref<?xi32>
// CHECK:               %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_1]]] : memref<?xi32> {
// CHECK:               ^bb0(%[[VAL_11:.*]]: i32):
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[VAL_11]], %[[EXTRACT_0]] : i32
// CHECK:                 %[[SELECT_0:.*]] = arith.select %[[EXTRACT_1]], %[[ADDI_1]], %[[VAL_11]] : i32
// CHECK:                 memref.atomic_yield %[[SELECT_0]] : i32
// CHECK:               }
// CHECK:               %[[INSERT_0:.*]] = tensor.insert %[[GENERIC_ATOMIC_RMW_0]] into %[[VAL_10]]{{\[}}%[[VAL_9]]] : tensor<4xi32>
// CHECK:               scf.yield %[[INSERT_0]] : tensor<4xi32>
// CHECK:             }
// CHECK:             %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel"]} ins(%[[VAL_5]], %[[FOR_1]] : tensor<4xi32>, tensor<4xi32>) outs(%[[VAL_5]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32):
// CHECK:               %[[ADDI_2:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:               linalg.yield %[[ADDI_2]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             scf.yield %[[GENERIC_3]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func public @atomic_rmw_tensor_ptr_masked(%arg0: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %init = arith.constant dense<0> : tensor<4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %mask = arith.cmpi slt, %range, %c2 : tensor<4xi32>
    %ptrs = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %ptrs2 = tt.addptr %ptrs, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %loop = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<4xi32>)  : i32 {
      %i_splat = tt.splat %i : i32 -> tensor<4xi32>
      %ptrs3 = tt.addptr %ptrs2, %i_splat : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
      %val = tt.atomic_rmw add, acq_rel, gpu, %ptrs3, %acc, %mask : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
      %sum = arith.addi %acc, %val : tensor<4xi32>
      scf.yield %sum : tensor<4xi32>
    }
    tt.return
  }
}
