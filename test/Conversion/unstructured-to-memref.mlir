// RUN: triton-shared-opt --split-input-file --unstructured-to-memref %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func public @scalar_gather_scatter(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           %[[LOAD_0:.*]] = affine.load %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:           affine.store %[[LOAD_0]], %[[REINTERPRET_CAST_1]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           tt.return
// CHECK:         }
  tt.func public @scalar_gather_scatter(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %off: i32) {
    %val = tts.gather %src[%off] : (<f32>, i32) -> f32
    tts.scatter %val into %dst[%off] : f32 into (<f32>, i32)
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @gather_no_mask(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[CAST_0]] restrict : memref<?xf32> to tensor<?xf32>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[CONSTANT_0]] : tensor<4xi32>) outs(%[[EMPTY_0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: f32):
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[INDEX_CAST_0]]] : tensor<?xf32>
// CHECK:             linalg.yield %[[EXTRACT_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           tt.return
// CHECK:         }
  tt.func public @gather_no_mask(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %val = tts.gather %src[%offsets] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @gather_mask_no_other(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[CAST_0]] restrict : memref<?xf32> to tensor<?xf32>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[CONSTANT_0]], %[[CONSTANT_1]] : tensor<4xi32>, tensor<4xi1>) outs(%[[EMPTY_0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: f32):
// CHECK:             %[[IF_0:.*]] = scf.if %[[VAL_1]] -> (f32) {
// CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[INDEX_CAST_0]]] : tensor<?xf32>
// CHECK:               scf.yield %[[EXTRACT_0]] : f32
// CHECK:             } else {
// CHECK:               %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:               scf.yield %[[CONSTANT_2]] : f32
// CHECK:             }
// CHECK:             linalg.yield %[[IF_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           tt.return
// CHECK:         }
  tt.func public @gather_mask_no_other(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %val = tts.gather %src[%offsets] mask = %mask : (<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @gather_mask_with_other(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1.250000e+00 : f32
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[CAST_0]] restrict : memref<?xf32> to tensor<?xf32>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel"]} ins(%[[CONSTANT_0]], %[[CONSTANT_1]] : tensor<4xi32>, tensor<4xi1>) outs(%[[EMPTY_0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i1, %[[VAL_2:.*]]: f32):
// CHECK:             %[[IF_0:.*]] = scf.if %[[VAL_1]] -> (f32) {
// CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[TO_TENSOR_0]]{{\[}}%[[INDEX_CAST_0]]] : tensor<?xf32>
// CHECK:               scf.yield %[[EXTRACT_0]] : f32
// CHECK:             } else {
// CHECK:               scf.yield %[[CONSTANT_2]] : f32
// CHECK:             }
// CHECK:             linalg.yield %[[IF_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           tt.return
// CHECK:         }
  tt.func public @gather_mask_with_other(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %other = arith.constant 1.250000e+00 : f32
    %val = tts.gather %src[%offsets] mask = %mask default = %other : (<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @scatter_no_mask(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_3]]], iterator_types = ["parallel"]} ins(%[[CONSTANT_0]], %[[CONSTANT_1]] : tensor<4xi32>, tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: f32):
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             memref.store %[[VAL_1]], %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @scatter_no_mask(%dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %values = arith.constant dense<1.000000e+00> : tensor<4xf32>
    tts.scatter %values into %dst[%offsets] : tensor<4xf32> into (<f32>, tensor<4xi32>)
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @scatter_mask(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[CONSTANT_0]], %[[CONSTANT_1]], %[[CONSTANT_2]] : tensor<4xi32>, tensor<4xf32>, tensor<4xi1>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: i1):
// CHECK:             scf.if %[[VAL_2]] {
// CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:               memref.store %[[VAL_1]], %[[CAST_0]]{{\[}}%[[INDEX_CAST_0]]] : memref<?xf32>
// CHECK:             }
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @scatter_mask(%dst: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %values = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    tts.scatter %values into %dst[%offsets] mask = %mask : tensor<4xf32> into (<f32>, tensor<4xi32>)
    tt.return
  }
}
