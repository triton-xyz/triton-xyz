// RUN: triton-shared-opt --split-input-file --triton-unstructured-fallback %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func public @masked_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]]) -> (tensor<4xf32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[SPLAT_0]]{{\[}}%[[VAL_0]]] : tensor<4x!tt.ptr<f32>>
// CHECK:             %[[LOAD_0:.*]] = tt.load %[[EXTRACT_0]], %[[CONSTANT_1]], %[[CONSTANT_0]] : !tt.ptr<f32>
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[LOAD_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]]] : tensor<4xf32>
// CHECK:             scf.yield %[[INSERT_0]] : tensor<4xf32>
// CHECK:           }
// CHECK:           scf.for %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[SPLAT_1]]{{\[}}%[[VAL_2]]] : tensor<4x!tt.ptr<f32>>
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[FOR_0]]{{\[}}%[[VAL_2]]] : tensor<4xf32>
// CHECK:             tt.store %[[EXTRACT_1]], %[[EXTRACT_2]], %[[CONSTANT_1]] : !tt.ptr<f32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @masked_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c0 = arith.constant 0.0 : f32
    %true = arith.constant true
    %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst_ptrs = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %mask = tt.splat %true : i1 -> tensor<4xi1>
    %other = tt.splat %c0 : f32 -> tensor<4xf32>
    %val = tt.load %src_ptrs, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.store %dst_ptrs, %val, %mask : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @tensor_atomic_rmw(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant true
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]]) -> (tensor<4xf32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[SPLAT_0]]{{\[}}%[[VAL_0]]] : tensor<4x!tt.ptr<f32>>
// CHECK:             %[[ATOMIC_RMW_0:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %[[EXTRACT_0]], %[[CONSTANT_1]], %[[CONSTANT_0]] : (!tt.ptr<f32>, f32, i1) -> f32
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[ATOMIC_RMW_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]]] : tensor<4xf32>
// CHECK:             scf.yield %[[INSERT_0]] : tensor<4xf32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @tensor_atomic_rmw(%ptr: !tt.ptr<f32>) {
    %c1 = arith.constant 1.0 : f32
    %true = arith.constant true
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %vals = tt.splat %c1 : f32 -> tensor<4xf32>
    %mask = tt.splat %true : i1 -> tensor<4xi1>
    %old = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %vals, %mask : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @tensor_atomic_cas(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_1]] step %[[CONSTANT_2]] iter_args(%[[VAL_1:.*]] = %[[EMPTY_0]]) -> (tensor<4xi32>) {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[SPLAT_0]]{{\[}}%[[VAL_0]]] : tensor<4x!tt.ptr<i32>>
// CHECK:             %[[ATOMIC_CAS_0:.*]] = tt.atomic_cas acq_rel, gpu, %[[EXTRACT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]] : (!tt.ptr<i32>, i32, i32) -> i32
// CHECK:             %[[INSERT_0:.*]] = tensor.insert %[[ATOMIC_CAS_0]] into %[[VAL_1]]{{\[}}%[[VAL_0]]] : tensor<4xi32>
// CHECK:             scf.yield %[[INSERT_0]] : tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @tensor_atomic_cas(%ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %ptrs = tt.splat %ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %cmp = tt.splat %c0 : i32 -> tensor<4xi32>
    %val = tt.splat %c0 : i32 -> tensor<4xi32>
    %old = tt.atomic_cas acq_rel, gpu, %ptrs, %cmp, %val : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    tt.return
  }
}
