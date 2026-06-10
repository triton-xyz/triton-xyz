// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @block_ptr_indirect_dim0_load_store(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<f16>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: !tt.ptr<f16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[3, 1, 0, 2]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4x4xf16>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[EXTRACT_0]], %[[CONSTANT_3]] : index
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[MULI_0]]], sizes: [1, 4], strides: [4, 1] : memref<*xf16> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]], 0] [1, 4] [1, 1] : memref<4x4xf16> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1x4xf16, strided<[4, 1], offset: ?>> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4x4xf16> to tensor<4x4xf16>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[EXTRACT_1]], %[[CONSTANT_3]] : index
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 4], strides: [4, 1] : memref<*xf16> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]], 0] [1, 4] [1, 1] : tensor<4x4xf16> to tensor<1x4xf16>
// CHECK:             bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1x4xf16>, memref<1x4xf16, strided<[4, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_ptr_indirect_dim0_load_store(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
    %idx = arith.constant dense<[3, 1, 0, 2]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], wrap_boundaries: [0, 0], layout: "block", parent_shape: [4, 4] {layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    %val = "tta.load"(%src_idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>

    %dst_addr = tta.make_addr %dst to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], wrap_boundaries: [0, 0], layout: "block", parent_shape: [4, 4] {layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>, tensor<4x4xf16>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_ptr_indirect_dim1_load_store(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<f16>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: !tt.ptr<f16>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[2, 0, 3, 1]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4x4xf16>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<4xindex>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_0]]], sizes: [4, 1], strides: [4, 1] : memref<*xf16> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]][0, %[[VAL_0]]] [4, 1] [1, 1] : memref<4x4xf16> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<4x1xf16, strided<[4, 1], offset: ?>> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4x4xf16> to tensor<4x4xf16>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<4xindex>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [4, 1], strides: [4, 1] : memref<*xf16> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]][0, %[[VAL_1]]] [4, 1] [1, 1] : tensor<4x4xf16> to tensor<4x1xf16>
// CHECK:             bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4x1xf16>, memref<4x1xf16, strided<[4, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_ptr_indirect_dim1_load_store(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
    %idx = arith.constant dense<[2, 0, 3, 1]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], wrap_boundaries: [0, 0], layout: "block", parent_shape: [4, 4] {layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    %val = "tta.load"(%src_idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>

    %dst_addr = tta.make_addr %dst to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], wrap_boundaries: [0, 0], layout: "block", parent_shape: [4, 4] {layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>, tensor<4x4xf16>) -> ()
    tt.return
  }
}
