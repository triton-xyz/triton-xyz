// RUN: triton-xyz-opt --split-input-file --tta-address-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @structured_basic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1]>> to memref<4xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @structured_basic(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_ptr_with_advance(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4x4xf16>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [1], sizes: [4, 4], strides: [4, 1] : memref<*xf16> to memref<4x4xf16, strided<[4, 1], offset: 1>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4x4xf16, strided<[4, 1], offset: 1>> to memref<4x4xf16>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4x4xf16> to tensor<4x4xf16>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [4, 4], strides: [4, 1] : memref<*xf16> to memref<4x4xf16, strided<[4, 1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4x4xf16>, memref<4x4xf16, strided<[4, 1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_ptr_with_advance(%base: !tt.ptr<f16>) {
    %addr = tta.make_addr %base to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], layout: [4, 4] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %next = "tta.advance"(%addr) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f16, 2, 1>) -> !tta.addr<f16, 2, 1>
    %val = "tta.load"(%next) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>
    "tta.store"(%addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>, tensor<4x4xf16>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @indirect_masked_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1.250000e+00 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_5]] : f32) outs(%[[ALLOC_0]] : memref<4xf32>)
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_4]]{{\[}}%[[VAL_0]]] : tensor<4xi1>
// CHECK:             scf.if %[[EXTRACT_0]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[CONSTANT_4]]{{\[}}%[[VAL_1]]] : tensor<4xi1>
// CHECK:             scf.if %[[EXTRACT_2]] {
// CHECK:               %[[EXTRACT_3:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[EXTRACT_3]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @indirect_masked_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %other = arith.constant 1.250000e+00 : f32

    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %offsets, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_idx, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %offsets, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @indirect_reindex_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_4]]{{\[}}%[[VAL_0]]] : tensor<4xi1>
// CHECK:             scf.if %[[EXTRACT_0]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[CONSTANT_4]]{{\[}}%[[VAL_1]]] : tensor<4xi1>
// CHECK:             scf.if %[[EXTRACT_2]] {
// CHECK:               %[[EXTRACT_3:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[EXTRACT_3]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @indirect_reindex_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>

    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %offsets, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %offsets, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}


// -----

module {
// CHECK-LABEL:   tt.func @imported_addr_basic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1]>> to memref<4xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @imported_addr_basic(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}


// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG2]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             %[[VAL_3:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_4:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_3]], %[[VAL_4]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}


// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_non_unit(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG2]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[VAL_0]], %[[CONSTANT_0]] : i32
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : i32
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[DIVSI_0]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             %[[VAL_3:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_4:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_3]], %[[VAL_4]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_non_unit(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c3 = arith.constant 3 : i32
    %c2 = arith.constant 2 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c3 to %n step %c2 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}


// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_index_iv(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 3 : i64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[ARG2]] step %[[CONSTANT_3]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : index to i64
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[INDEX_CAST_0]], %[[CONSTANT_1]] : i64
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_0]] : i64
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[DIVSI_0]] : i64 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             %[[VAL_3:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_4:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_3]], %[[VAL_4]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_index_iv(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: index) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c3 to %n step %c2 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}


// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_reindex(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG2]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             %[[VAL_3:.*]] = "tta.reindex"(%[[VAL_1]]) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_4:.*]] = "tta.reindex"(%[[VAL_2]]) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_3]], %[[VAL_4]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_reindex(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.reindex"(%src_addr) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.reindex"(%dst_addr) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_derived_use(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[C1:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[ARG2]] step %[[C1]] iter_args(%[[SRC_ADDR:.*]] = %[[MAKE_ADDR_0]], %[[DST_ADDR:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[IV_IDX_0:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:             %[[SRC_OFF:.*]] = arith.addi %[[IV_IDX_0]], %[[SRC_CONST:.*]] : index
// CHECK:             %[[SRC_BASE:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[SRC_ALLOC:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[SRC_VIEW:.*]] = memref.reinterpret_cast %[[SRC_BASE]] to offset: {{\[}}%[[SRC_OFF]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[SRC_VIEW]], %[[SRC_ALLOC]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK:             %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[SRC_ALLOC]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[DST_OFF:.*]] = arith.addi %[[IV_IDX_0]], %[[DST_CONST:.*]] : index
// CHECK:             %[[DST_BASE:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[DST_VIEW:.*]] = memref.reinterpret_cast %[[DST_BASE]] to offset: {{\[}}%[[DST_OFF]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR]] in writable %[[DST_VIEW]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             %[[NEXT_SRC:.*]] = "tta.advance"(%[[SRC_ADDR]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[NEXT_DST:.*]] = "tta.reindex"(%[[DST_ADDR]]) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[NEXT_SRC]], %[[NEXT_DST]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_derived_use(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %src_use = "tta.reindex"(%src_addr) <{static_offsets = array<i64: 2>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %dst_use = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 3>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %val = "tta.load"(%src_use) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_use, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.reindex"(%dst_addr) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_seed(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[IDX:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %{{.*}} to %[[ARG2]] step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[IV_IDX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:             scf.for %[[INNER_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %[[EXTRACT:.*]] = tensor.extract %[[IDX]]{{\[}}%[[INNER_IV]]] : tensor<4xindex>
// CHECK:               %[[OFFSET:.*]] = arith.addi %[[EXTRACT]], %[[IV_IDX]] : index
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_seed(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>

    %src_base = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_base = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %src_addr0 = "tta.indirect_reindex"(%src_base, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
    %dst_addr0 = "tta.indirect_reindex"(%dst_base, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>

    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.reindex"(%dst_addr) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[IDX:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %{{.*}} to %[[ARG2]] step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[IV_IDX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:             %[[SPLAT:.*]] = tensor.splat %[[IV_IDX]] : tensor<4xindex>
// CHECK:             %[[SCALED:.*]] = arith.muli %[[SPLAT]], %[[IDX]] : tensor<4xindex>
// CHECK:             %[[RECUR:.*]] = arith.addi %[[SCALED]], %[[IDX]] : tensor<4xindex>
// CHECK:             scf.for %[[INNER_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %[[EXTRACT:.*]] = tensor.extract %[[RECUR]]{{\[}}%[[INNER_IV]]] : tensor<4xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>

    %src_base = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %src_addr0 = "tta.indirect_reindex"(%src_base, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>

    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>

    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.indirect_reindex"(%src_addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[IDX:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[IDX_I32:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[FOR:.*]]:2 = scf.for %[[IV:.*]] = %{{.*}} to %[[ARG2]] step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[IV_IDX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:             %[[IS_ZERO:.*]] = arith.cmpi eq, %[[IV_IDX]], %{{.*}} : index
// CHECK:             %[[SPLAT:.*]] = tensor.splat %[[IV_IDX]] : tensor<4xindex>
// CHECK:             %[[SCALED:.*]] = arith.muli %[[SPLAT]], %[[IDX]] : tensor<4xindex>
// CHECK:             %[[CUR_IDX:.*]] = arith.select %[[IS_ZERO]], %[[IDX]], %[[SCALED]] : tensor<4xindex>
// CHECK:             scf.for %[[INNER_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %[[EXTRACT:.*]] = tensor.extract %[[CUR_IDX]]{{\[}}%[[INNER_IV]]] : tensor<4xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %out0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %addr0, %out = %out0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%out, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next = "tta.indirect_reindex"(%addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      %next_out = "tta.advance"(%out) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next, %next_out : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed_dynamic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi1>,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[IDX_CAST:.*]] = arith.index_cast %[[ARG2]] : tensor<?xi32> to tensor<?xindex>
// CHECK:           %[[IDX_DIM:.*]] = tensor.dim %[[IDX_CAST]], %{{.*}} : tensor<?xindex>
// CHECK:           %[[IDENTITY:.*]] = tensor.generate %[[IDX_DIM]]
// CHECK:           %[[MASK_DIM:.*]] = tensor.dim %[[ARG3]], %{{.*}} : tensor<?xi1>
// CHECK:           %[[ALL_TRUE:.*]] = tensor.generate %[[MASK_DIM]]
// CHECK:           scf.if %{{.*}} {
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed_dynamic(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %idx: tensor<?xi32>, %mask: tensor<?xi1>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %addr0 = tta.make_addr %src to sizes: [4], strides: [2], offsets: [3], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %out0 = tta.make_addr %dst to sizes: [4], strides: [2], offsets: [5], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %addr0, %out = %out0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%out, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next = "tta.indirect_reindex"(%addr, %idx, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 1, 1>
      %next_out = "tta.advance"(%out) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next, %next_out : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_mixed_seed(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[IV_IDX:.*]] = arith.index_cast %[[IV:.*]] : i32 to index
// CHECK:           tensor.splat %[[IV_IDX]] : tensor<2xindex>
// CHECK:           tensor.splat %[[IV_IDX]] : tensor<4xindex>
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_mixed_seed(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant dense<[0, 1]> : tensor<2xi32>
    %idx1 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>

    %src_base = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_seed = "tta.indirect_reindex"(%src_base, %idx0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>

    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_seed, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
      %next_src0 = "tta.indirect_reindex"(%src_addr, %idx0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
      %next_src1 = "tta.indirect_reindex"(%next_src0, %idx1) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
      scf.yield %next_src1, %next_dst : !tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_dynamic_wrap(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi1>,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           arith.index_cast %[[ARG2]] : tensor<?xi32> to tensor<?xindex>
// CHECK:           tensor.dim %{{.*}}, %{{.*}} : tensor<?xindex>
// CHECK:           tensor.dim %[[ARG3]], %{{.*}} : tensor<?xi1>
// CHECK:           arith.andi
// CHECK:           arith.remsi
// CHECK:           arith.select
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_dynamic_wrap(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %idx_dyn: tensor<?xi32>, %mask_dyn: tensor<?xi1>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask0 = arith.constant dense<[true, false]> : tensor<2xi1>

    %src_base = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [1, 3], layout: [5, 9] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_seed = "tta.indirect_reindex"(%src_base, %idx0, %mask0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 2], layout: [7, 11] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>

    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_seed, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
      %next_src0 = "tta.indirect_reindex"(%src_addr, %idx0, %mask0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
      %next_src1 = "tta.indirect_reindex"(%next_src0, %idx_dyn, %mask_dyn) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 2, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
      scf.yield %next_src1, %next_dst : !tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_same_dim_step_merge(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           tensor.splat %{{.*}} : tensor<2xindex>
// CHECK:           arith.addi %{{.*}}, %{{.*}} : tensor<2xindex>
// CHECK:           tensor.splat %{{.*}} : tensor<4xindex>
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_same_dim_step_merge(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0_a = arith.constant dense<[0, 1]> : tensor<2xi32>
    %idx0_b = arith.constant dense<[1, 0]> : tensor<2xi32>
    %idx1 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask0_a = arith.constant dense<[true, false]> : tensor<2xi1>
    %mask0_b = arith.constant dense<[true, true]> : tensor<2xi1>

    %src_base = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_seed = "tta.indirect_reindex"(%src_base, %idx0_a, %mask0_a) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>

    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%src_addr = %src_seed, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
      %next_src0 = "tta.indirect_reindex"(%src_addr, %idx0_a, %mask0_a) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
      %next_src1 = "tta.indirect_reindex"(%next_src0, %idx0_b, %mask0_b) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
      %next_src2 = "tta.indirect_reindex"(%next_src1, %idx1) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
      scf.yield %next_src2, %next_dst : !tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @indirect_non_zero_offset(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[EXTRACT_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[ADDI_0]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[EXTRACT_1]], %[[CONSTANT_1]] : index
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[ADDI_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:             bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @indirect_non_zero_offset(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %src_off = "tta.reindex"(%src_addr) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    %src_idx = "tta.indirect_reindex"(%src_off, %offsets) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%src_idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %dst_off = "tta.reindex"(%dst_addr) <{static_offsets = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_off, %offsets) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @chained_indirect_same_dim(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi1>,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi1>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : tensor<4xi32> to tensor<4xindex>
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : tensor<4xi32> to tensor<4xindex>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[INDEX_CAST_1]] : tensor<4xindex>
// CHECK:           %[[ANDI_0:.*]] = arith.andi %[[ARG3]], %[[ARG4]] : tensor<4xi1>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ANDI_0]]{{\[}}%[[VAL_0]]] : tensor<4xi1>
// CHECK:             scf.if %[[EXTRACT_0]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[ADDI_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[ANDI_0]]{{\[}}%[[VAL_1]]] : tensor<4xi1>
// CHECK:             scf.if %[[EXTRACT_2]] {
// CHECK:               %[[EXTRACT_3:.*]] = tensor.extract %[[ADDI_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_3]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @chained_indirect_same_dim(%src: !tt.ptr<f32>, %idx0: tensor<4xi32>, %idx1: tensor<4xi32>, %mask0: tensor<4xi1>, %mask1: tensor<4xi1>) {
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %r0 = "tta.indirect_reindex"(%addr, %idx0, %mask0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %r1 = "tta.indirect_reindex"(%r0, %idx1, %mask1) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%r1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %out_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %out_r0 = "tta.indirect_reindex"(%out_addr, %idx0, %mask0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %out_r1 = "tta.indirect_reindex"(%out_r0, %idx1, %mask1) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%out_r1, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @chained_indirect_dynamic_static_dim(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi1>,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi1>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : tensor<?xi32> to tensor<?xindex>
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : tensor<4xi32> to tensor<4xindex>
// CHECK:           %[[CAST_0:.*]] = tensor.cast %[[INDEX_CAST_1]] : tensor<4xindex> to tensor<?xindex>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CAST_0]] : tensor<?xindex>
// CHECK:           %[[CAST_1:.*]] = tensor.cast %[[ARG4]] : tensor<4xi1> to tensor<?xi1>
// CHECK:           %[[ANDI_0:.*]] = arith.andi %[[ARG3]], %[[CAST_1]] : tensor<?xi1>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[DIM_0:.*]] = tensor.dim %[[ADDI_0]], %[[CONSTANT_2]] : tensor<?xindex>
// CHECK:           %[[MINSI_0:.*]] = arith.minsi %[[DIM_0]], %[[CONSTANT_1]] : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[MINSI_0]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ANDI_0]]{{\[}}%[[VAL_0]]] : tensor<?xi1>
// CHECK:             scf.if %[[EXTRACT_0]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[ADDI_0]]{{\[}}%[[VAL_0]]] : tensor<?xindex>
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[MINSI_0]] step %[[CONSTANT_0]] {
// CHECK:             %[[EXTRACT_2:.*]] = tensor.extract %[[ANDI_0]]{{\[}}%[[VAL_1]]] : tensor<?xi1>
// CHECK:             scf.if %[[EXTRACT_2]] {
// CHECK:               %[[EXTRACT_3:.*]] = tensor.extract %[[ADDI_0]]{{\[}}%[[VAL_1]]] : tensor<?xindex>
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_3]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @chained_indirect_dynamic_static_dim(%src: !tt.ptr<f32>, %idx_dyn: tensor<?xi32>, %idx_static: tensor<4xi32>, %mask_dyn: tensor<?xi1>, %mask_static: tensor<4xi1>) {
    %addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %r0 = "tta.indirect_reindex"(%addr, %idx_dyn, %mask_dyn) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 1, 1>
    %r1 = "tta.indirect_reindex"(%r0, %idx_static, %mask_static) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    %val = "tta.load"(%r1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>

    %out_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %out_r0 = "tta.indirect_reindex"(%out_addr, %idx_dyn, %mask_dyn) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 1, 1>
    %out_r1 = "tta.indirect_reindex"(%out_r0, %idx_static, %mask_static) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 1, 1>
    "tta.store"(%out_r1, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @indirect_reindex_dim1_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_4]] : f32) outs(%[[ALLOC_0]] : memref<2x4xf32>)
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_0]]], sizes: [2, 1], strides: [4, 1] : memref<*xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]][0, %[[VAL_0]]] [2, 1] [1, 1] : memref<2x4xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<2x1xf32, strided<[4, 1], offset: ?>> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [2, 1], strides: [4, 1] : memref<*xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:             %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]][0, %[[VAL_1]]] [2, 1] [1, 1] : tensor<2x4xf32> to tensor<2x1xf32>
// CHECK:             bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<2x1xf32>, memref<2x1xf32, strided<[4, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @indirect_reindex_dim1_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %idx = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_idx, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @indirect_reindex_multi_dim_load_store(
// CHECK:           %[[IDX1:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[IDX0:.*]] = arith.constant dense<[0, 1]> : tensor<2xindex>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[IV0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:             scf.for %[[IV1:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:               %[[R0:.*]] = tensor.extract %[[IDX0]]{{\[}}%[[IV0]]] : tensor<2xindex>
// CHECK:               %[[R0M:.*]] = arith.muli %[[R0]], %[[C4]] : index
// CHECK:               %[[R1:.*]] = tensor.extract %[[IDX1]]{{\[}}%[[IV1]]] : tensor<4xindex>
// CHECK:               %[[LIN0:.*]] = arith.addi %[[R0M]], %[[R1]] : index
// CHECK:               %[[V:.*]] = memref.load %{{.*}}{{\[}}%{{.*}}] : memref<?xf32>
// CHECK:               memref.store %[[V]], %{{.*}}{{\[}}%[[IV0]], %[[IV1]]] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           scf.for %[[IV2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:             scf.for %[[IV3:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:               %[[S0:.*]] = tensor.extract %[[IDX0]]{{\[}}%[[IV2]]] : tensor<2xindex>
// CHECK:               %[[S0M:.*]] = arith.muli %[[S0]], %[[C4]] : index
// CHECK:               %[[S1:.*]] = tensor.extract %[[IDX1]]{{\[}}%[[IV3]]] : tensor<4xindex>
// CHECK:               %[[LIN1:.*]] = arith.addi %[[S0M]], %[[S1]] : index
// CHECK:               %[[E:.*]] = tensor.extract %{{.*}}{{\[}}%[[IV2]], %[[IV3]]] : tensor<2x4xf32>
// CHECK:               memref.store %[[E]], %{{.*}}{{\[}}%{{.*}}] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @indirect_reindex_multi_dim_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %offsets0 = arith.constant dense<[0, 1]> : tensor<2xi32>
    %offsets1 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_idx0 = "tta.indirect_reindex"(%src_addr, %offsets0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
    %src_idx1 = "tta.indirect_reindex"(%src_idx0, %offsets1) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    %val = "tta.load"(%src_idx1) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %dst_idx0 = "tta.indirect_reindex"(%dst_addr, %offsets0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
    %dst_idx1 = "tta.indirect_reindex"(%dst_idx0, %offsets1) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
    "tta.store"(%dst_idx1, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_multi_indirect_load_store(
// CHECK:           %[[MASK0:.*]] = arith.constant dense<[true, false]> : tensor<2xi1>
// CHECK:           %[[MASK1:.*]] = arith.constant dense<[true, true, false, true]> : tensor<4xi1>
// CHECK:           scf.for %[[IV0:.*]] = %[[C0:.*]] to %[[C2:.*]] step %[[C1:.*]] {
// CHECK:             scf.for %[[IV1:.*]] = %[[C0]] to %[[C4:.*]] step %[[C1]] {
// CHECK:               %[[M0:.*]] = tensor.extract %[[MASK0]]{{\[}}%[[IV0]]] : tensor<2xi1>
// CHECK:               %[[M1:.*]] = tensor.extract %[[MASK1]]{{\[}}%[[IV1]]] : tensor<4xi1>
// CHECK:               %[[M:.*]] = arith.andi %[[M0]], %[[M1]] : i1
// CHECK:               scf.if %[[M]] {
// CHECK:                 %[[R:.*]] = arith.remsi {{.*}} : index
// CHECK:                 %[[S:.*]] = arith.select {{.*}} : index
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_multi_indirect_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %idx0 = arith.constant dense<[0, 1]> : tensor<2xi32>
    %idx1 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask0 = arith.constant dense<[true, false]> : tensor<2xi1>
    %mask1 = arith.constant dense<[true, true, false, true]> : tensor<4xi1>

    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 1], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_idx0 = "tta.indirect_reindex"(%src_addr, %idx0, %mask0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %src_idx1 = "tta.indirect_reindex"(%src_idx0, %idx1, %mask1) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 2, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_idx1, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 2], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %dst_idx0 = "tta.indirect_reindex"(%dst_addr, %idx0, %mask0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %dst_idx1 = "tta.indirect_reindex"(%dst_idx0, %idx1, %mask1) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>, tensor<4xi1>) -> !tta.addr<f32, 2, 1>
    "tta.store"(%dst_idx1, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_boundary_no_wrap_fastpath(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [1], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: 1>>
// CHECK:           linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[ALLOC_0]] : memref<4xf32>)
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1], offset: 1>> to memref<4xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [2], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: 2>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: 2>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_no_wrap_fastpath(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [1], layout: [16] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_addr, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [2], layout: [16] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_boundary_rank1_segmented_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [6], sizes: [2], strides: [1] : memref<*xf32> to memref<2xf32, strided<[1], offset: 6>>
// CHECK:           %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]][0] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<2xf32, strided<[1], offset: 6>> to memref<2xf32, strided<[1]>>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [2], strides: [1] : memref<*xf32> to memref<2xf32, strided<[1]>>
// CHECK:           %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_0]][2] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 2>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_1]], %[[SUBVIEW_1]] : memref<2xf32, strided<[1]>> to memref<2xf32, strided<[1], offset: 2>>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [6], sizes: [2], strides: [1] : memref<*xf32> to memref<2xf32, strided<[1], offset: 6>>
// CHECK:           %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]][0] [2] [1] : tensor<4xf32> to tensor<2xf32>
// CHECK:           bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_2]] : (tensor<2xf32>, memref<2xf32, strided<[1], offset: 6>>) -> ()
// CHECK:           %[[REINTERPRET_CAST_3:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [2], strides: [1] : memref<*xf32> to memref<2xf32, strided<[1]>>
// CHECK:           %[[EXTRACT_SLICE_1:.*]] = tensor.extract_slice %[[TO_TENSOR_0]][2] [2] [1] : tensor<4xf32> to tensor<2xf32>
// CHECK:           bufferization.materialize_in_destination %[[EXTRACT_SLICE_1]] in writable %[[REINTERPRET_CAST_3]] : (tensor<2xf32>, memref<2xf32, strided<[1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_rank1_segmented_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [6], layout: [8] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_addr, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [6], layout: [8] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_boundary_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_5]] : f32) outs(%[[ALLOC_0]] : memref<2x4xf32>)
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] {
// CHECK:             scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:               %[[MULI_0:.*]] = arith.muli %[[VAL_0]], %[[CONSTANT_3]] : index
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK:               %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_3]] : index
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_2]] : index
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_3]] : index
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:               %[[ADDI_2:.*]] = arith.addi %[[MULI_0]], %[[SELECT_0]] : index
// CHECK:               %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[ADDI_2]]] : memref<?xf32>
// CHECK:               memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]], %[[VAL_1]]] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_1]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] {
// CHECK:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:               %[[MULI_1:.*]] = arith.muli %[[VAL_2]], %[[CONSTANT_3]] : index
// CHECK:               %[[ADDI_3:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_1]] : index
// CHECK:               %[[REMSI_1:.*]] = arith.remsi %[[ADDI_3]], %[[CONSTANT_3]] : index
// CHECK:               %[[CMPI_1:.*]] = arith.cmpi slt, %[[REMSI_1]], %[[CONSTANT_2]] : index
// CHECK:               %[[ADDI_4:.*]] = arith.addi %[[REMSI_1]], %[[CONSTANT_3]] : index
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[ADDI_4]], %[[REMSI_1]] : index
// CHECK:               %[[ADDI_5:.*]] = arith.addi %[[MULI_1]], %[[SELECT_1]] : index
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : tensor<2x4xf32>
// CHECK:               memref.store %[[EXTRACT_0]], %[[CAST_1]]{{\[}}%[[ADDI_5]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 3], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_addr, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 1], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_boundary_mixed_indirect_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1]> : tensor<2xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 3 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant dense<[true, false]> : tensor<2xi1>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_6]] : f32) outs(%[[ALLOC_0]] : memref<2x4xf32>)
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_5]] step %[[CONSTANT_2]] {
// CHECK:             scf.for %[[VAL_1:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_7]]{{\[}}%[[VAL_0]]] : tensor<2xi1>
// CHECK:               scf.if %[[EXTRACT_0]] {
// CHECK:                 %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<2xindex>
// CHECK:                 %[[MULI_0:.*]] = arith.muli %[[EXTRACT_1]], %[[CONSTANT_4]] : index
// CHECK:                 %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_1]] : index
// CHECK:                 %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_4]] : index
// CHECK:                 %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_3]] : index
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_4]] : index
// CHECK:                 %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:                 %[[ADDI_2:.*]] = arith.addi %[[MULI_0]], %[[SELECT_0]] : index
// CHECK:                 %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[ADDI_2]]] : memref<?xf32>
// CHECK:                 memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]], %[[VAL_1]]] : memref<2x4xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_1]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_2:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_5]] step %[[CONSTANT_2]] {
// CHECK:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] {
// CHECK:               %[[EXTRACT_2:.*]] = tensor.extract %[[CONSTANT_7]]{{\[}}%[[VAL_2]]] : tensor<2xi1>
// CHECK:               scf.if %[[EXTRACT_2]] {
// CHECK:                 %[[EXTRACT_3:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_2]]] : tensor<2xindex>
// CHECK:                 %[[MULI_1:.*]] = arith.muli %[[EXTRACT_3]], %[[CONSTANT_4]] : index
// CHECK:                 %[[ADDI_3:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_2]] : index
// CHECK:                 %[[REMSI_1:.*]] = arith.remsi %[[ADDI_3]], %[[CONSTANT_4]] : index
// CHECK:                 %[[CMPI_1:.*]] = arith.cmpi slt, %[[REMSI_1]], %[[CONSTANT_3]] : index
// CHECK:                 %[[ADDI_4:.*]] = arith.addi %[[REMSI_1]], %[[CONSTANT_4]] : index
// CHECK:                 %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[ADDI_4]], %[[REMSI_1]] : index
// CHECK:                 %[[ADDI_5:.*]] = arith.addi %[[MULI_1]], %[[SELECT_1]] : index
// CHECK:                 %[[EXTRACT_4:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : tensor<2x4xf32>
// CHECK:                 memref.store %[[EXTRACT_4]], %[[CAST_1]]{{\[}}%[[ADDI_5]]] : memref<?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_mixed_indirect_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>

    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 3], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %idx, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_idx, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 1], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %idx, %mask) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_boundary_negative_offset_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -3 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant -1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_6]] : f32) outs(%[[ALLOC_0]] : memref<2x4xf32>)
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_5]] step %[[CONSTANT_2]] {
// CHECK:             scf.for %[[VAL_1:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] {
// CHECK:               %[[MULI_0:.*]] = arith.muli %[[VAL_0]], %[[CONSTANT_4]] : index
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_1]] : index
// CHECK:               %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_4]] : index
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_3]] : index
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_4]] : index
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:               %[[ADDI_2:.*]] = arith.addi %[[MULI_0]], %[[SELECT_0]] : index
// CHECK:               %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[ADDI_2]]] : memref<?xf32>
// CHECK:               memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]], %[[VAL_1]]] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_1]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_2:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_5]] step %[[CONSTANT_2]] {
// CHECK:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_2]] {
// CHECK:               %[[MULI_1:.*]] = arith.muli %[[VAL_2]], %[[CONSTANT_4]] : index
// CHECK:               %[[ADDI_3:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK:               %[[REMSI_1:.*]] = arith.remsi %[[ADDI_3]], %[[CONSTANT_4]] : index
// CHECK:               %[[CMPI_1:.*]] = arith.cmpi slt, %[[REMSI_1]], %[[CONSTANT_3]] : index
// CHECK:               %[[ADDI_4:.*]] = arith.addi %[[REMSI_1]], %[[CONSTANT_4]] : index
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[ADDI_4]], %[[REMSI_1]] : index
// CHECK:               %[[ADDI_5:.*]] = arith.addi %[[MULI_1]], %[[SELECT_1]] : index
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : tensor<2x4xf32>
// CHECK:               memref.store %[[EXTRACT_0]], %[[CAST_1]]{{\[}}%[[ADDI_5]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_negative_offset_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, -1], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_addr, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, f32) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, -3], layout: [0, 4] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_boundary_dynamic_guard(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_4]] : f32) outs(%[[ALLOC_0]] : memref<4xf32>)
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi sgt, %[[ARG2]], %[[CONSTANT_2]] : index
// CHECK:             cf.assert %[[CMPI_0]], "tta-to-memref: wrap boundary must be > 0"
// CHECK:             %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[ARG2]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[ARG2]] : index
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xf32>
// CHECK:             memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_0]]] : memref<4xf32>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_1]] : memref<*xf32> to memref<?xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi sgt, %[[ARG2]], %[[CONSTANT_2]] : index
// CHECK:             cf.assert %[[CMPI_2]], "tta-to-memref: wrap boundary must be > 0"
// CHECK:             %[[REMSI_1:.*]] = arith.remsi %[[ADDI_2]], %[[ARG2]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi slt, %[[REMSI_1]], %[[CONSTANT_2]] : index
// CHECK:             %[[ADDI_3:.*]] = arith.addi %[[REMSI_1]], %[[ARG2]] : index
// CHECK:             %[[SELECT_1:.*]] = arith.select %[[CMPI_3]], %[[ADDI_3]], %[[REMSI_1]] : index
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]]] : tensor<4xf32>
// CHECK:             memref.store %[[EXTRACT_0]], %[[CAST_1]]{{\[}}%[[SELECT_1]]] : memref<?xf32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_dynamic_guard(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %boundary: index) {
    %src_addr = tta.make_addr %src to sizes: [4], strides: [1], offsets: [1], layout: [%boundary] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %other = arith.constant 0.0 : f32
    %val = "tta.load"(%src_addr, %other) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, f32) -> tensor<4xf32>
    %dst_addr = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [2], layout: [%boundary] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_scalar_basic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[ARG3]], %[[ADDI_0]], %[[VAL_0]] : i32
// CHECK:             memref.atomic_yield %[[SELECT_0]] : i32
// CHECK:           }
// CHECK:           %[[GENERIC_ATOMIC_RMW_1:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32):
// CHECK:             memref.atomic_yield %[[GENERIC_ATOMIC_RMW_0]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_scalar_basic(%ptr: !tt.ptr<i32>, %off: i32, %val: i32, %mask: i1) {
    %ptr_i = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r0 = "tta.atomic"(%ptr_i, %off, %val, %mask) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32, i1) -> i32
    %r1 = "tta.atomic"(%ptr_i, %off, %r0) <{kind = "xchg"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %r2 = arith.addi %r0, %r1 : i32
    %r3 = arith.addi %r2, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_scalar_wrap_boundary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             memref.atomic_yield %[[ADDI_2]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_scalar_wrap_boundary(%ptr: !tt.ptr<i32>, %off: i32, %val: i32) {
    %ptr_i = tta.make_addr %ptr to sizes: [16], strides: [1], offsets: [3], layout: [8] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_scalar_wrap_negative_boundary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 8 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant -5 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             memref.atomic_yield %[[ADDI_2]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_scalar_wrap_negative_boundary(%ptr: !tt.ptr<i32>, %off: i32, %val: i32) {
    %ptr_i = tta.make_addr %ptr to sizes: [16], strides: [1], offsets: [-5], layout: [8] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_scalar_wrap_dynamic_boundary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[ARG3]], %[[CONSTANT_0]] : index
// CHECK:           cf.assert %[[CMPI_0]], "tta-to-memref: wrap boundary must be > 0"
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[ARG3]] : index
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[ARG3]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             memref.atomic_yield %[[ADDI_2]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_scalar_wrap_dynamic_boundary(%ptr: !tt.ptr<i32>, %off: i32, %val: i32, %boundary: index) {
    %ptr_i = tta.make_addr %ptr to sizes: [16], strides: [1], offsets: [2], layout: [%boundary] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val) <{kind = "add"}> : (!tta.addr<i32, 1, 1>, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_float_add(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32):
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_0]], %[[ARG2]] : f32
// CHECK:             memref.atomic_yield %[[ADDF_0]] : f32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_float_add(%ptr: !tt.ptr<f32>, %off: i32, %val: f32) {
    %ptr_i = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %r = "tta.atomic"(%ptr_i, %off, %val) <{kind = "fadd"}> : (!tta.addr<f32, 1, 1>, i32, f32) -> f32
    %u = arith.addf %r, %val : f32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_cas_scalar_wrap_boundary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 16 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 5 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[ARG3]], %[[VAL_0]] : i32
// CHECK:             memref.atomic_yield %[[SELECT_1]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_cas_scalar_wrap_boundary(%ptr: !tt.ptr<i32>, %off: i32, %cmp: i32, %val: i32) {
    %ptr_i = tta.make_addr %ptr to sizes: [32], strides: [1], offsets: [5], layout: [16] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %off, %cmp, %val) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_cas_scalar_wrap_negative_boundary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 16 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant -7 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[ARG3]], %[[VAL_0]] : i32
// CHECK:             memref.atomic_yield %[[SELECT_1]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_cas_scalar_wrap_negative_boundary(%ptr: !tt.ptr<i32>, %off: i32, %cmp: i32, %val: i32) {
    %ptr_i = tta.make_addr %ptr to sizes: [32], strides: [1], offsets: [-7], layout: [16] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %off, %cmp, %val) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_cas_scalar_wrap_dynamic_boundary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_1]] : index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[ARG4]], %[[CONSTANT_0]] : index
// CHECK:           cf.assert %[[CMPI_0]], "tta-to-memref: wrap boundary must be > 0"
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[ADDI_0]], %[[ARG4]] : index
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi slt, %[[REMSI_0]], %[[CONSTANT_0]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[REMSI_0]], %[[ARG4]] : index
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_1]], %[[ADDI_1]], %[[REMSI_0]] : index
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[SELECT_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi eq, %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             %[[SELECT_1:.*]] = arith.select %[[CMPI_2]], %[[ARG3]], %[[VAL_0]] : i32
// CHECK:             memref.atomic_yield %[[SELECT_1]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_cas_scalar_wrap_dynamic_boundary(%ptr: !tt.ptr<i32>, %off: i32, %cmp: i32, %val: i32, %boundary: index) {
    %ptr_i = tta.make_addr %ptr to sizes: [32], strides: [1], offsets: [4], layout: [%boundary] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %off, %cmp, %val) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_cas_scalar(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xi32> to memref<?xi32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xi32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[ARG3]], %[[VAL_0]] : i32
// CHECK:             memref.atomic_yield %[[SELECT_0]] : i32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_cas_scalar(%ptr: !tt.ptr<i32>, %off: i32, %cmp: i32, %val: i32) {
    %ptr_i = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <i32> to !tta.addr<i32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %off, %cmp, %val) : (!tta.addr<i32, 1, 1>, i32, i32, i32) -> i32
    %u = arith.addi %r, %val : i32
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @atomic_cas_float(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[GENERIC_ATOMIC_RMW_0:.*]] = memref.generic_atomic_rmw %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32):
// CHECK:             %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[ARG2]] : f32
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[ARG3]], %[[VAL_0]] : f32
// CHECK:             memref.atomic_yield %[[SELECT_0]] : f32
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @atomic_cas_float(%ptr: !tt.ptr<f32>, %off: i32, %cmp: f32, %val: f32) {
    %ptr_i = tta.make_addr %ptr to sizes: [1], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %r = "tta.atomic_cas"(%ptr_i, %off, %cmp, %val) : (!tta.addr<f32, 1, 1>, i32, f32, f32) -> f32
    %u = arith.addf %r, %val : f32
    tt.return
  }
}
