// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @wrap_boundary_row_fastpath(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [4], sizes: [1, 4], strides: [4, 1] : memref<*xf32> to memref<1x4xf32, strided<[4, 1], offset: 4>>
// CHECK:           %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]][0, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<1x4xf32, strided<[4, 1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1x4xf32, strided<[4, 1], offset: 4>> to memref<1x4xf32, strided<[4, 1]>>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [1, 4], strides: [4, 1] : memref<*xf32> to memref<1x4xf32, strided<[4, 1]>>
// CHECK:           %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_0]][1, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<1x4xf32, strided<[4, 1], offset: 4>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_1]], %[[SUBVIEW_1]] : memref<1x4xf32, strided<[4, 1]>> to memref<1x4xf32, strided<[4, 1], offset: 4>>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [2, 4], strides: [4, 1] : memref<*xf32> to memref<2x4xf32, strided<[4, 1]>>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_2]] : memref<2x4xf32, strided<[4, 1]>> to memref<2x4xf32, strided<[?, 1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[CAST_0]] : (tensor<2x4xf32>, memref<2x4xf32, strided<[?, 1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_row_fastpath(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c4 = arith.constant 4 : index
    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [%c4, 1], offsets: [%c4, 0], wrap_boundaries: [8, 0], layout: "strided" : <f32> to !tta.addr<f32, 2, 1>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [%c4, 1], offsets: [0, 0], wrap_boundaries: [0, 0], layout: "strided" : <f32> to !tta.addr<f32, 2, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_boundary_row_store_fastpath(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [2, 4], strides: [4, 1] : memref<*xf32> to memref<2x4xf32, strided<[4, 1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<2x4xf32, strided<[4, 1]>> to memref<2x4xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]][0, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [4], sizes: [1, 4], strides: [4, 1] : memref<*xf32> to memref<1x4xf32, strided<[4, 1], offset: 4>>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<1x4xf32, strided<[4, 1], offset: 4>> to memref<1x4xf32, strided<[?, 1], offset: 4>>
// CHECK:           bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[CAST_0]] : (tensor<1x4xf32>, memref<1x4xf32, strided<[?, 1], offset: 4>>) -> ()
// CHECK:           %[[EXTRACT_SLICE_1:.*]] = tensor.extract_slice %[[TO_TENSOR_0]][1, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
// CHECK:           %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [1, 4], strides: [4, 1] : memref<*xf32> to memref<1x4xf32, strided<[4, 1]>>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_2]] : memref<1x4xf32, strided<[4, 1]>> to memref<1x4xf32, strided<[?, 1]>>
// CHECK:           bufferization.materialize_in_destination %[[EXTRACT_SLICE_1]] in writable %[[CAST_1]] : (tensor<1x4xf32>, memref<1x4xf32, strided<[?, 1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @wrap_boundary_row_store_fastpath(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %c4 = arith.constant 4 : index
    %src_addr = tta.make_addr %src to sizes: [2, 4], strides: [%c4, 1], offsets: [0, 0], wrap_boundaries: [0, 0], layout: "strided" : <f32> to !tta.addr<f32, 2, 1>
    %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>

    %dst_addr = tta.make_addr %dst to sizes: [2, 4], strides: [%c4, 1], offsets: [%c4, 0], wrap_boundaries: [8, 0], layout: "strided" : <f32> to !tta.addr<f32, 2, 1>
    "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
    tt.return
  }
}
