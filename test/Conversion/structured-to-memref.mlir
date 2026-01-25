// RUN: triton-shared-opt --structured-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @structured_basic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1]>> to memref<4xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @structured_basic(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %src_tptr = tts.make_tptr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %val = "tts.load"(%src_tptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    %dst_tptr = tts.make_tptr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    "tts.store"(%dst_tptr, %val) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @masked_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.250000e+00 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[ALLOC_0]] : memref<4xf32>)
// CHECK:           %[[SUBVIEW_0:.*]] = memref.subview %[[REINTERPRET_CAST_0]][0] [2] [1] : memref<4xf32, strided<[1]>> to memref<2xf32, strided<[1]>>
// CHECK:           %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_0]][0] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1]>>
// CHECK:           memref.copy %[[SUBVIEW_0]], %[[SUBVIEW_1]] : memref<2xf32, strided<[1]>> to memref<2xf32, strided<[1]>>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]][0] [2] [1] : tensor<4xf32> to tensor<2xf32>
// CHECK:           %[[SUBVIEW_2:.*]] = memref.subview %[[REINTERPRET_CAST_1]][0] [2] [1] : memref<4xf32, strided<[1]>> to memref<2xf32, strided<[1]>>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[SUBVIEW_2]] : memref<2xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[CAST_0]] : (tensor<2xf32>, memref<?xf32, strided<[1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @masked_load_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %other = arith.constant 1.250000e+00 : f32
    %mask = arith.constant 2 : index
    %src_tptr = tts.make_tptr %src to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    %val = "tts.load"(%src_tptr, %mask, %other) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<4x!tt.ptr<f32>>, index, f32) -> tensor<4xf32>
    %dst_tptr = tts.make_tptr %dst to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
    "tts.store"(%dst_tptr, %val, %mask) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, index) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @block_ptr_basic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f16>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f16> to memref<*xf16>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [2, 3], strides: [3, 1] : memref<*xf16> to memref<2x3xf16, strided<[3, 1]>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x3xf16>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<2x3xf16, strided<[3, 1]>> to memref<2x3xf16>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x3xf16> to tensor<2x3xf16>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_0]] : (tensor<2x3xf16>, memref<2x3xf16, strided<[3, 1]>>) -> ()
// CHECK:           tt.return
// CHECK:         }
  tt.func @block_ptr_basic(%base: !tt.ptr<f16>) {
    %ptr = tts.make_tptr %base to sizes: [2, 3], strides: [3, 1], offsets: [0, 0], shape: [2, 3], order: [1, 0] : <f16> to !tt.ptr<tensor<2x3xf16>>
    %val = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<2x3xf16>>) -> tensor<2x3xf16>
    "tts.store"(%ptr, %val) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<2x3xf16>>, tensor<2x3xf16>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @gather_scatter_load_store(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1]> : tensor<2xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x2xf32>
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_0]]] : tensor<2xindex>
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[EXTRACT_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[MULI_0]]], sizes: [1, 2], strides: [2, 1] : memref<*xf32> to memref<1x2xf32, strided<[2, 1], offset: ?>>
// CHECK:             %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_0]], 0] [1, 2] [1, 1] : memref<2x2xf32> to memref<1x2xf32, strided<[2, 1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1x2xf32, strided<[2, 1], offset: ?>> to memref<1x2xf32, strided<[2, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x2xf32> to tensor<2x2xf32>
// CHECK:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK:             %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_1]]] : tensor<2xindex>
// CHECK:             %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_1]], 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<1x2xf32>
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[EXTRACT_1]], %[[CONSTANT_2]] : index
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 2], strides: [2, 1] : memref<*xf32> to memref<1x2xf32, strided<[2, 1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1x2xf32>, memref<1x2xf32, strided<[2, 1], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @gather_scatter_load_store(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1]> : tensor<2xi32>
    %tptr = tts.make_gather_scatter_tptr %src to sizes: [2, 2] gather_scatter_dim: 0 gather_scatter_offset: %offsets, strides: [2, 1], offsets: [0, 0] : tensor<2xi32>  <f32> to !tt.ptr<tensor<2x2xf32>>
    %val = "tts.load"(%tptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<2x2xf32>>) -> tensor<2x2xf32>
    "tts.store"(%tptr, %val) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<2x2xf32>>, tensor<2x2xf32>) -> ()
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_side_by_side(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> tensor<2x4xf32> {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [3], sizes: [2, 1], strides: [4, 1] : memref<*xf32> to memref<2x1xf32, strided<[4, 1], offset: 3>>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [2, 3], strides: [4, 1] : memref<*xf32> to memref<2x3xf32, strided<[4, 1]>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]][0, 0] [2, 1] [1, 1] : memref<2x4xf32> to memref<2x1xf32, strided<[4, 1]>>
// CHECK:           %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_0]][0, 1] [2, 3] [1, 1] : memref<2x4xf32> to memref<2x3xf32, strided<[4, 1], offset: 1>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<2x1xf32, strided<[4, 1], offset: 3>> to memref<2x1xf32, strided<[4, 1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_1]], %[[SUBVIEW_1]] : memref<2x3xf32, strided<[4, 1]>> to memref<2x3xf32, strided<[4, 1], offset: 1>>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:           tt.return %[[TO_TENSOR_0]] : tensor<2x4xf32>
// CHECK:         }
  tt.func @wrap_side_by_side(%src: !tt.ptr<f32>) -> tensor<2x4xf32> {
    %tptr = tts.make_tptr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 3], shape: [0, 4], order: [] : <f32> to tensor<2x4x!tt.ptr<f32>>
    %val = "tts.load"(%tptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x4x!tt.ptr<f32>>) -> tensor<2x4xf32>
    tt.return %val : tensor<2x4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @wrap_stacked(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> tensor<2x2xf32> {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [2], sizes: [1, 2], strides: [2, 1] : memref<*xf32> to memref<1x2xf32, strided<[2, 1], offset: 2>>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: [0], sizes: [1, 2], strides: [2, 1] : memref<*xf32> to memref<1x2xf32, strided<[2, 1]>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x2xf32>
// CHECK:           %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]][0, 0] [1, 2] [1, 1] : memref<2x2xf32> to memref<1x2xf32, strided<[2, 1]>>
// CHECK:           %[[SUBVIEW_1:.*]] = memref.subview %[[ALLOC_0]][1, 0] [1, 2] [1, 1] : memref<2x2xf32> to memref<1x2xf32, strided<[2, 1], offset: 2>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1x2xf32, strided<[2, 1], offset: 2>> to memref<1x2xf32, strided<[2, 1]>>
// CHECK:           memref.copy %[[REINTERPRET_CAST_1]], %[[SUBVIEW_1]] : memref<1x2xf32, strided<[2, 1]>> to memref<1x2xf32, strided<[2, 1], offset: 2>>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x2xf32> to tensor<2x2xf32>
// CHECK:           tt.return %[[TO_TENSOR_0]] : tensor<2x2xf32>
// CHECK:         }
  tt.func @wrap_stacked(%src: !tt.ptr<f32>) -> tensor<2x2xf32> {
    %tptr = tts.make_tptr %src to sizes: [2, 2], strides: [2, 1], offsets: [2, 0], shape: [4, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
    %val = "tts.load"(%tptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
    tt.return %val : tensor<2x2xf32>
  }
}
