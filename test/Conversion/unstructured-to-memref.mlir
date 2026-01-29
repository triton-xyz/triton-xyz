// RUN: triton-xyz-opt --split-input-file --unstructured-to-memref %s | FileCheck %s

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
    %val = tts.gather %src[%off] : (!tt.ptr<f32>, i32) -> f32
    tts.scatter %val into %dst[%off] : f32 into (!tt.ptr<f32>, i32)
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @gather_no_mask(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           tt.return
// CHECK:         }
  tt.func public @gather_no_mask(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %val = tts.gather %src[%offsets] : (!tt.ptr<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @gather_mask_no_other(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           tt.return
// CHECK:         }
  tt.func public @gather_mask_no_other(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %val = tts.gather %src[%offsets] mask = %mask : (!tt.ptr<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @gather_mask_with_other(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           tt.return
// CHECK:         }
  tt.func public @gather_mask_with_other(%src: !tt.ptr<f32>) {
    %offsets = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %mask = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
    %other = arith.constant 1.250000e+00 : f32
    %val = tts.gather %src[%offsets] mask = %mask default = %other : (!tt.ptr<f32>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @scatter_no_mask(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[CONSTANT_0]], %[[CONSTANT_1]] : tensor<4xi32>, tensor<4xf32>) {
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
    tts.scatter %values into %dst[%offsets] : tensor<4xf32> into (!tt.ptr<f32>, tensor<4xi32>)
    tt.return
  }
}

// -----

module {
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @scatter_mask(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[CONSTANT_0]], %[[CONSTANT_1]], %[[CONSTANT_2]] : tensor<4xi32>, tensor<4xf32>, tensor<4xi1>) {
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
    tts.scatter %values into %dst[%offsets] mask = %mask : tensor<4xf32> into (!tt.ptr<f32>, tensor<4xi32>)
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @multi_base_gather(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4x!tt.ptr<f32>>) {
// CHECK:           tt.return
// CHECK:         }
  tt.func public @multi_base_gather(%ptrs: tensor<4x!tt.ptr<f32>>) {
    %c0 = arith.constant 0 : i32
    %offs = tt.splat %c0 : i32 -> tensor<4xi32>
    %val = tts.gather %ptrs[%offs] : (tensor<4x!tt.ptr<f32>>, tensor<4xi32>) -> tensor<4xf32>
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func public @multi_base_scatter(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4x!tt.ptr<f32>>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] {
// CHECK:             %[[EXTRACT_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_0]]] : tensor<4x!tt.ptr<f32>>
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[EXTRACT_0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[CONSTANT_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store %[[CONSTANT_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @multi_base_scatter(%ptrs: tensor<4x!tt.ptr<f32>>) {
    %c0 = arith.constant 0 : i32
    %offs = tt.splat %c0 : i32 -> tensor<4xi32>
    %vals = arith.constant dense<1.000000e+00> : tensor<4xf32>
    tts.scatter %vals into %ptrs[%offs] : tensor<4xf32> into (tensor<4x!tt.ptr<f32>>, tensor<4xi32>)
    tt.return
  }
}
