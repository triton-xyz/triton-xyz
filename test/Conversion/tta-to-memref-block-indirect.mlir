// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @block_ptr_indirect_dim0_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[IDX:.*]] = arith.constant dense<[3, 1, 0, 2]> : tensor<4xindex>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<4x4xf16>
// CHECK:           scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:             %[[OFF:.*]] = tensor.extract %[[IDX]]{{\[}}%[[IV]]] : tensor<4xindex>
// CHECK:             %[[OFF4:.*]] = arith.muli %[[OFF]], %[[C4]] : index
// CHECK:             %[[SRC:.*]] = memref.reinterpret_cast %[[SRC_CAST]] to offset: {{\[}}%[[OFF4]]], sizes: [1, 4], strides: [4, 1] : memref<*xf16> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[DST:.*]] = memref.subview %[[ALLOC]]{{\[}}%[[IV]], 0] [1, 4] [1, 1] : memref<4x4xf16> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[SRC]], %[[DST]] : memref<1x4xf16, strided<[4, 1], offset: ?>> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4x4xf16> to tensor<4x4xf16>
// CHECK:           %[[DST_CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           scf.for %[[IV2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:             %[[OFF2:.*]] = tensor.extract %[[IDX]]{{\[}}%[[IV2]]] : tensor<4xindex>
// CHECK:             %[[OFF24:.*]] = arith.muli %[[OFF2]], %[[C4]] : index
// CHECK:             %[[DST2:.*]] = memref.reinterpret_cast %[[DST_CAST]] to offset: {{\[}}%[[OFF24]]], sizes: [1, 4], strides: [4, 1] : memref<*xf16> to memref<1x4xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[SLICE:.*]] = tensor.extract_slice %[[TENSOR]]{{\[}}%[[IV2]], 0] [1, 4] [1, 1] : tensor<4x4xf16> to tensor<1x4xf16>
// CHECK:             bufferization.materialize_in_destination %[[SLICE]] in writable %[[DST2]] : (tensor<1x4xf16>, memref<1x4xf16, strided<[4, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_ptr_indirect_dim0_load_store(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
    %idx = arith.constant dense<[3, 1, 0, 2]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], layout: [4, 4] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    %val = "tta.load"(%src_idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>

    %dst_addr = tta.make_addr %dst to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], layout: [4, 4] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>, tensor<4x4xf16>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_ptr_indirect_dim1_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[IDX:.*]] = arith.constant dense<[2, 0, 3, 1]> : tensor<4xindex>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[SRC_CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<4x4xf16>
// CHECK:           scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:             %[[OFF:.*]] = tensor.extract %[[IDX]]{{\[}}%[[IV]]] : tensor<4xindex>
// CHECK:             %[[SRC:.*]] = memref.reinterpret_cast %[[SRC_CAST]] to offset: {{\[}}%[[OFF]]], sizes: [4, 1], strides: [4, 1] : memref<*xf16> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[DST:.*]] = memref.subview %[[ALLOC]][0, %[[IV]]] [4, 1] [1, 1] : memref<4x4xf16> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:             memref.copy %[[SRC]], %[[DST]] : memref<4x1xf16, strided<[4, 1], offset: ?>> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<4x4xf16> to tensor<4x4xf16>
// CHECK:           %[[DST_CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           scf.for %[[IV2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:             %[[OFF2:.*]] = tensor.extract %[[IDX]]{{\[}}%[[IV2]]] : tensor<4xindex>
// CHECK:             %[[DST2:.*]] = memref.reinterpret_cast %[[DST_CAST]] to offset: {{\[}}%[[OFF2]]], sizes: [4, 1], strides: [4, 1] : memref<*xf16> to memref<4x1xf16, strided<[4, 1], offset: ?>>
// CHECK:             %[[SLICE:.*]] = tensor.extract_slice %[[TENSOR]][0, %[[IV2]]] [4, 1] [1, 1] : tensor<4x4xf16> to tensor<4x1xf16>
// CHECK:             bufferization.materialize_in_destination %[[SLICE]] in writable %[[DST2]] : (tensor<4x1xf16>, memref<4x1xf16, strided<[4, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_ptr_indirect_dim1_load_store(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
    %idx = arith.constant dense<[2, 0, 3, 1]> : tensor<4xi32>

    %src_addr = tta.make_addr %src to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], layout: [4, 4] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %src_idx = "tta.indirect_reindex"(%src_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    %val = "tta.load"(%src_idx) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>) -> tensor<4x4xf16>

    %dst_addr = tta.make_addr %dst to sizes: [4, 4], strides: [4, 1], offsets: [0, 0], layout: [4, 4] {layout_kind = "block", layout_payload = {order = array<i32: 1, 0>}} : <f16> to !tta.addr<f16, 2, 1>
    %dst_idx = "tta.indirect_reindex"(%dst_addr, %idx) <{indirect_dim = 1 : i32}> : (!tta.addr<f16, 2, 1>, tensor<4xi32>) -> !tta.addr<f16, 2, 1>
    "tta.store"(%dst_idx, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f16, 2, 1>, tensor<4x4xf16>) -> ()
    tt.return
  }
}
