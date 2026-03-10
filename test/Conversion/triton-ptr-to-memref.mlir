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
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[CST:.*]] = arith.constant 1.000000e+00 : f16
// CHECK:           %[[CAST:.*]] = memref.cast %[[ARG0]] : memref<*xf16> to memref<?xf16>
// CHECK:           memref.store %[[CST]], %[[CAST]]{{\[}}%[[C0]]] : memref<?xf16>
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
// CHECK-LABEL:   func.func @scalar_ptr_if(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK:           %[[SRC_CAST0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %[[SRC_CAST0]] : memref<?xf32> -> index
// CHECK:           %[[ADDR:.*]] = arith.index_cast %[[BASE]] : index to i64
// CHECK:           %[[COND:.*]] = arith.cmpi ne, %[[ADDR]], %[[C0_I64]] : i64
// CHECK:           scf.if %[[COND]] {
// CHECK:             %[[IDX:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:             %[[SRC:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:             %[[VAL:.*]] = memref.load %[[SRC]]{{\[}}%[[IDX]]] : memref<?xf32>
// CHECK:             %[[IDX2:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:             %[[DST:.*]] = memref.cast %[[ARG1]] : memref<*xf32> to memref<?xf32>
// CHECK:             memref.store %[[VAL]], %[[DST]]{{\[}}%[[IDX2]]] : memref<?xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
  func.func @scalar_ptr_if(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
    %1 = arith.cmpi ne, %0, %c0_i64 : i64
    scf.if %1 {
      %2 = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
      %3 = tt.load %2 : !tt.ptr<f32>
      %4 = tt.addptr %arg1, %arg2 : !tt.ptr<f32>, i32
      tt.store %4, %3 : !tt.ptr<f32>
    }
    return
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

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_addptr_atomic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[CONSTANT_0]]] : tensor<4xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG2]] -> (f32) {
// CHECK:             %[[ATOMIC_0:.*]] = memref.atomic_rmw addf %[[ARG3]], %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : (f32, memref<?xf32>) -> f32
// CHECK:             scf.yield %[[ATOMIC_0]] : f32
// CHECK:           } else {
// CHECK:             scf.yield %[[LOAD_0]] : f32
// CHECK:           }
// CHECK:           tt.return %[[IF_0]] : f32
// CHECK:         }
  tt.func @tensor_ptr_addptr_atomic(%arg0: !tt.ptr<f32>, %arg1: tensor<4xi32>, %arg2: i1, %arg3: f32) -> f32 {
    %base = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %base_ptr = builtin.unrealized_conversion_cast %base : memref<*xf32> to !tt.ptr<f32>
    %empty = tensor.empty() : tensor<4x!tt.ptr<f32>>
    %ptrs = linalg.fill ins(%base_ptr : !tt.ptr<f32>) outs(%empty : tensor<4x!tt.ptr<f32>>) -> tensor<4x!tt.ptr<f32>>
    %offset_ptrs = tt.addptr %ptrs, %arg1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %c0 = arith.constant 0 : index
    %ptr = tensor.extract %offset_ptrs[%c0] : tensor<4x!tt.ptr<f32>>
    %result = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %arg3, %arg2 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return %result : f32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_fill_from_scalar_addptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>) -> f32 {
// CHECK:           %[[C5:.*]] = arith.constant 5 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[C0]]] : tensor<4xi32>
// CHECK:           %[[OFFSET_IDX:.*]] = arith.index_cast %[[EXTRACT]] : i32 to index
// CHECK:           %[[CAST:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[LOAD_IDX:.*]] = arith.addi %[[OFFSET_IDX]], %[[C5]] : index
// CHECK:           %[[VAL:.*]] = memref.load %[[CAST]]{{\[}}%[[LOAD_IDX]]] : memref<?xf32>
// CHECK:           tt.return %[[VAL]] : f32
// CHECK:         }
  tt.func @tensor_ptr_fill_from_scalar_addptr(%arg0: !tt.ptr<f32>, %arg1: tensor<4xi32>) -> f32 {
    %c5_i32 = arith.constant 5 : i32
    %base = tt.addptr %arg0, %c5_i32 : !tt.ptr<f32>, i32
    %empty = tensor.empty() : tensor<4x!tt.ptr<f32>>
    %ptrs = linalg.fill ins(%base : !tt.ptr<f32>) outs(%empty : tensor<4x!tt.ptr<f32>>) -> tensor<4x!tt.ptr<f32>>
    %offset_ptrs = tt.addptr %ptrs, %arg1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %c0 = arith.constant 0 : index
    %ptr = tensor.extract %offset_ptrs[%c0] : tensor<4x!tt.ptr<f32>>
    %val = tt.load %ptr : !tt.ptr<f32>
    tt.return %val : f32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_addptr_atomic_cas_float(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[CONSTANT_0]]] : tensor<4xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
// CHECK:           %[[BITCAST_0:.*]] = arith.bitcast %[[LOAD_0]] : f32 to i32
// CHECK:           %[[BITCAST_1:.*]] = arith.bitcast %[[ARG2]] : f32 to i32
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[BITCAST_0]], %[[BITCAST_1]] : i32
// CHECK:           scf.if %[[CMPI_0]] {
// CHECK:             memref.store %[[ARG3]], %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
// CHECK:           }
// CHECK:           tt.return %[[LOAD_0]] : f32
// CHECK:         }
  tt.func @tensor_ptr_addptr_atomic_cas_float(%arg0: !tt.ptr<f32>, %arg1: tensor<4xi32>, %arg2: f32, %arg3: f32) -> f32 {
    %base = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %base_ptr = builtin.unrealized_conversion_cast %base : memref<*xf32> to !tt.ptr<f32>
    %empty = tensor.empty() : tensor<4x!tt.ptr<f32>>
    %ptrs = linalg.fill ins(%base_ptr : !tt.ptr<f32>) outs(%empty : tensor<4x!tt.ptr<f32>>) -> tensor<4x!tt.ptr<f32>>
    %offset_ptrs = tt.addptr %ptrs, %arg1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %c0 = arith.constant 0 : index
    %ptr = tensor.extract %offset_ptrs[%c0] : tensor<4x!tt.ptr<f32>>
    %result = tt.atomic_cas acq_rel, gpu, %ptr, %arg2, %arg3 : (!tt.ptr<f32>, f32, f32) -> f32
    tt.return %result : f32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_addptr_atomic_max_bitcast_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[CONSTANT_0]]] : tensor<4xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[BITCAST_0:.*]] = arith.bitcast %[[ARG3]] : i32 to f32
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
// CHECK:           %[[BITCAST_1:.*]] = arith.bitcast %[[LOAD_0]] : f32 to i32
// CHECK:           %[[IF_0:.*]] = scf.if %[[ARG2]] -> (i32) {
// CHECK:             %[[ATOMIC_0:.*]] = memref.atomic_rmw maximumf %[[BITCAST_0]], %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : (f32, memref<?xf32>) -> f32
// CHECK:             %[[BITCAST_2:.*]] = arith.bitcast %[[ATOMIC_0]] : f32 to i32
// CHECK:             scf.yield %[[BITCAST_2]] : i32
// CHECK:           } else {
// CHECK:             scf.yield %[[BITCAST_1]] : i32
// CHECK:           }
// CHECK:           tt.return %[[IF_0]] : i32
// CHECK:         }
  tt.func @tensor_ptr_addptr_atomic_max_bitcast_ptr(%arg0: !tt.ptr<f32>, %arg1: tensor<4xi32>, %arg2: i1, %arg3: i32) -> i32 {
    %base = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %base_ptr = builtin.unrealized_conversion_cast %base : memref<*xf32> to !tt.ptr<f32>
    %bitcast_ptr = tt.bitcast %base_ptr : !tt.ptr<f32> -> !tt.ptr<i32>
    %empty = tensor.empty() : tensor<4x!tt.ptr<i32>>
    %ptrs = linalg.fill ins(%bitcast_ptr : !tt.ptr<i32>) outs(%empty : tensor<4x!tt.ptr<i32>>) -> tensor<4x!tt.ptr<i32>>
    %offset_ptrs = tt.addptr %ptrs, %arg1 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %c0 = arith.constant 0 : index
    %ptr = tensor.extract %offset_ptrs[%c0] : tensor<4x!tt.ptr<i32>>
    %result = tt.atomic_rmw max, acq_rel, gpu, %ptr, %arg3, %arg2 : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return %result : i32
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @tensor_ptr_addptr_atomic_umin_bitcast_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[CONSTANT_0]]] : tensor<4xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[BITCAST_0:.*]] = arith.bitcast %[[ARG2]] : i32 to f32
// CHECK:           %[[ATOMIC_0:.*]] = memref.atomic_rmw maximumf %[[BITCAST_0]], %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : (f32, memref<?xf32>) -> f32
// CHECK:           %[[BITCAST_1:.*]] = arith.bitcast %[[ATOMIC_0]] : f32 to i32
// CHECK:           tt.return %[[BITCAST_1]] : i32
// CHECK:         }
  tt.func @tensor_ptr_addptr_atomic_umin_bitcast_ptr(%arg0: !tt.ptr<f32>, %arg1: tensor<4xi32>, %arg2: i32) -> i32 {
    %base = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %base_ptr = builtin.unrealized_conversion_cast %base : memref<*xf32> to !tt.ptr<f32>
    %bitcast_ptr = tt.bitcast %base_ptr : !tt.ptr<f32> -> !tt.ptr<i32>
    %empty = tensor.empty() : tensor<4x!tt.ptr<i32>>
    %ptrs = linalg.fill ins(%bitcast_ptr : !tt.ptr<i32>) outs(%empty : tensor<4x!tt.ptr<i32>>) -> tensor<4x!tt.ptr<i32>>
    %offset_ptrs = tt.addptr %ptrs, %arg1 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %c0 = arith.constant 0 : index
    %ptr = tensor.extract %offset_ptrs[%c0] : tensor<4x!tt.ptr<i32>>
    %result = tt.atomic_rmw umin, acq_rel, gpu, %ptr, %arg2 : (!tt.ptr<i32>, i32) -> i32
    tt.return %result : i32
  }
}

// -----

module {
// CHECK-LABEL:   func.func @scalar_i1_ptr_bitcast_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xi8>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK-NOT:       tt.bitcast
// CHECK-NOT:       !tt.ptr
// CHECK:           %[[CAST:.*]] = memref.cast %[[ARG0]] : memref<*xi8> to memref<?xi8>
// CHECK:           %[[EXT:.*]] = arith.extui %[[ARG1]] : i1 to i8
// CHECK:           memref.store %[[EXT]], %[[CAST]]
// CHECK:           return
  func.func @scalar_i1_ptr_bitcast_store(%arg0: !tt.ptr<i1>, %arg1: i1) {
    %base = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<i1> to memref<*xi8>
    %base_ptr = builtin.unrealized_conversion_cast %base : memref<*xi8> to !tt.ptr<i1>
    %bitcast_ptr = tt.bitcast %base_ptr : !tt.ptr<i1> -> !tt.ptr<i8>
    %target = builtin.unrealized_conversion_cast %bitcast_ptr : !tt.ptr<i8> to memref<*xi8>
    %cast = memref.cast %target : memref<*xi8> to memref<?xi8>
    %c0 = arith.constant 0 : index
    %ext = arith.extui %arg1 : i1 to i8
    memref.store %ext, %cast[%c0] : memref<?xi8>
    return
  }
}
