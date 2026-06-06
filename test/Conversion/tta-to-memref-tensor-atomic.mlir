// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @atomic_tensor_add(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi1>) {
// CHECK:           %[[ZERO_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]]) -> (tensor<4xi32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[ARG2]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[ARG3]]{{\[}}%[[VAL_0]]] : tensor<4xi1>
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:             %[[IF_0:.*]] = scf.if %[[EXTRACT_2]] -> (i32) {
// CHECK:               %[[ATOMIC_RMW_0:.*]] = memref.atomic_rmw addi %[[EXTRACT_1]], %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : (i32, memref<?xi32>) -> i32
// CHECK:               scf.yield %[[ATOMIC_RMW_0]] : i32
// CHECK:             } else {
// CHECK:               scf.yield %[[ZERO_I32]] : i32
// CHECK:             }
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[IF_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             scf.yield %[[INSERT_0]] : tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_tensor_add(%ptr: !tt.ptr<i32>, %off: tensor<4xi32>, %val: tensor<4xi32>, %mask: tensor<4xi1>) {
    %ptr_i = tta.make_addr %ptr to sizes: [16], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val, %mask) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, tensor<4xi32>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_tensor_cas(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]]) -> (tensor<4xi32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[ARG2]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[ARG3]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:             %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xi32> {
// CHECK:             ^bb0(%[[VAL_2:.*]]: i32):
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_2]], %[[EXTRACT_1]] : i32
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[EXTRACT_2]], %[[VAL_2]] : i32
// CHECK:               memref.atomic_yield %[[SELECT_0]] : i32
// CHECK:             }
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[GENERIC_ATOMIC_RMW_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             scf.yield %[[INSERT_0]] : tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_tensor_cas(%ptr: !tt.ptr<i32>, %off: tensor<4xi32>, %cmp: tensor<4xi32>, %val: tensor<4xi32>) {
    %ptr_i = tta.make_addr %ptr to sizes: [16], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %off, %cmp, %val) : (!tta.addr<i32, 1, 1>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_tensor_xchg_wrap(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi1>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]]) -> (tensor<4xi32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[ARG2]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[ARG3]]{{\[}}%[[VAL_0]]] : tensor<4xi1>
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_4]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:             %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xi32> {
// CHECK:             ^bb0(%[[VAL_2:.*]]: i32):
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[EXTRACT_2]], %[[EXTRACT_1]], %[[VAL_2]] : i32
// CHECK:               memref.atomic_yield %[[SELECT_1]] : i32
// CHECK:             }
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[GENERIC_ATOMIC_RMW_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             scf.yield %[[INSERT_0]] : tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_tensor_xchg_wrap(%ptr: !tt.ptr<i32>, %off: tensor<4xi32>, %val: tensor<4xi32>, %mask: tensor<4xi1>) {
    %ptr_i = tta.make_addr %ptr to sizes: [16], strides: [1], offsets: [3], wrap_boundaries: [8], layout: "strided" : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val, %mask) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, tensor<4xi32>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    tt.return
  }
}
