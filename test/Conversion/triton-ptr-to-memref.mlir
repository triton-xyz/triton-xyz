// RUN: triton-xyz-opt --split-input-file --triton-ptr-to-memref %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @func_ptr_args(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xi8>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           return
// CHECK:         }
  func.func @func_ptr_args(%arg0: !tt.ptr<f32>, %arg1: tensor<4x!tt.ptr<i8>>, %arg2: i32) {
    return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @ptr_select_to_memref_select(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[ARG2]], %[[ARG0]], %[[ARG1]] : memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[SELECT_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           memref.store %[[CONSTANT_0]], %[[REINTERPRET_CAST_0]]{{\[}}%[[CONSTANT_1]]] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           tt.return
// CHECK:         }
  tt.func @ptr_select_to_memref_select(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %pred: i1) {
    %c0_idx = arith.constant 0 : index
    %sel = arith.select %pred, %arg0, %arg1 : !tt.ptr<f32>
    %u = builtin.unrealized_conversion_cast %sel : !tt.ptr<f32> to memref<*xf32>
    %v = memref.reinterpret_cast %u to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %v[%c0_idx] : memref<1xf32, strided<[1], offset: ?>>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tt_ptr_arg(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<*xf16> to !tt.ptr<f16>
// CHECK:           tt.store %[[UNREALIZED_CONVERSION_CAST_0]], %[[CONSTANT_0]] : !tt.ptr<f16>
// CHECK:           tt.return
// CHECK:         }
  tt.func @tt_ptr_arg(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant 1.000000e+00 : f16
    tt.store %arg0, %cst : !tt.ptr<f16>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   func.func @callee(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           return %[[CONSTANT_0]] : f32
// CHECK:         }
  func.func @callee(%arg0: !tt.ptr<f32>) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    return %cst : f32
  }

// CHECK-LABEL:   func.func @caller(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) -> f32 {
// CHECK:           %[[VAL_0:.*]] = call @callee(%[[ARG0]]) : (memref<*xf32>) -> f32
// CHECK:           return %[[VAL_0]] : f32
// CHECK:         }
  func.func @caller(%arg0: !tt.ptr<f32>) -> f32 {
    %0 = func.call @callee(%arg0) : (!tt.ptr<f32>) -> f32
    return %0 : f32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[CONSTANT_0]]] : tensor<4xf32>
// CHECK:           memref.store %[[EXTRACT_0]], %[[ARG0]]{{\[}}%[[CONSTANT_0]]] : memref<4xf32>
// CHECK:           tt.return
// CHECK:         }
  tt.func @tensor_ptr_store(%arg0: tensor<4x!tt.ptr<f32>>, %arg1: tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %ptr = tensor.extract %arg0[%c0] : tensor<4x!tt.ptr<f32>>
    %val = tensor.extract %arg1[%c0] : tensor<4xf32>
    tt.store %ptr, %val : !tt.ptr<f32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_masked_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG2]] -> (f32) {
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[CONSTANT_0]]] : memref<4xf32>
// CHECK:             scf.yield %[[LOAD_0]] : f32
// CHECK:           } else {
// CHECK:             scf.yield %[[ARG3]] : f32
// CHECK:           }
// CHECK:           scf.if %[[ARG2]] {
// CHECK:             memref.store %[[IF_0]], %[[ARG1]]{{\[}}%[[CONSTANT_0]]] : memref<4xf32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @tensor_ptr_masked_load_store(%srcs: tensor<4x!tt.ptr<f32>>, %dsts: tensor<4x!tt.ptr<f32>>, %mask: i1, %other: f32) {
    %c0 = arith.constant 0 : index
    %src = tensor.extract %srcs[%c0] : tensor<4x!tt.ptr<f32>>
    %dst = tensor.extract %dsts[%c0] : tensor<4x!tt.ptr<f32>>
    %val = tt.load %src, %mask, %other : !tt.ptr<f32>
    tt.store %dst, %val, %mask : !tt.ptr<f32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_masked_load_no_other(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG1]] -> (f32) {
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[CONSTANT_1]]] : memref<4xf32>
// CHECK:             scf.yield %[[LOAD_0]] : f32
// CHECK:           } else {
// CHECK:             scf.yield %[[CONSTANT_0]] : f32
// CHECK:           }
// CHECK:           tt.return %[[IF_0]] : f32
// CHECK:         }
  tt.func @tensor_ptr_masked_load_no_other(%srcs: tensor<4x!tt.ptr<f32>>, %mask: i1) -> f32 {
    %c0 = arith.constant 0 : index
    %src = tensor.extract %srcs[%c0] : tensor<4x!tt.ptr<f32>>
    %val = tt.load %src, %mask : !tt.ptr<f32>
    tt.return %val : f32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_multi_dim(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x3xi16>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2x3xi16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_0]]] : tensor<2x3xi16>
// CHECK:           memref.store %[[EXTRACT_0]], %[[ARG0]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_0]]] : memref<2x3xi16>
// CHECK:           tt.return
// CHECK:         }
  tt.func @tensor_ptr_multi_dim(%ptrs: tensor<2x3x!tt.ptr<i16>>, %vals: tensor<2x3xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ptr = tensor.extract %ptrs[%c0, %c1] : tensor<2x3x!tt.ptr<i16>>
    %val = tensor.extract %vals[%c0, %c1] : tensor<2x3xi16>
    tt.store %ptr, %val : !tt.ptr<i16>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_dynamic_index(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) {
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG2]]{{\[}}%[[ARG3]]] : tensor<4xf32>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[ARG0]]{{\[}}%[[ARG3]]] : memref<4xf32>
// CHECK:           %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[EXTRACT_0]] : f32
// CHECK:           memref.store %[[ADDF_0]], %[[ARG1]]{{\[}}%[[ARG3]]] : memref<4xf32>
// CHECK:           tt.return
// CHECK:         }
  tt.func @tensor_ptr_dynamic_index(%srcs: tensor<4x!tt.ptr<f32>>, %dsts: tensor<4x!tt.ptr<f32>>, %vals: tensor<4xf32>, %idx: index) {
    %src = tensor.extract %srcs[%idx] : tensor<4x!tt.ptr<f32>>
    %dst = tensor.extract %dsts[%idx] : tensor<4x!tt.ptr<f32>>
    %val = tensor.extract %vals[%idx] : tensor<4xf32>
    %loaded = tt.load %src : !tt.ptr<f32>
    %sum = arith.addf %loaded, %val : f32
    tt.store %dst, %sum : !tt.ptr<f32>
    tt.return
  }
}
