// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @program_info(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i32 {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG3]], %[[ARG0]] : i32
// CHECK:           return %[[ADDI_0]] : i32
// CHECK:         }
  tt.func @program_info() -> i32 {
    %pid = tt.get_program_id x : i32
    %nprog = tt.get_num_programs x : i32
    %sum = arith.addi %pid, %nprog : i32
    tt.return %sum : i32
  }
}

// -----

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @broadcast_transpose(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4x2xf32> {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<2x4xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]] : tensor<1x4xi32>) outs(%[[EMPTY_1]] : tensor<2x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_1]] : i32
// CHECK:           } -> tensor<2x4xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_1]] : tensor<2x4xi32>) outs(%[[EMPTY_2]] : tensor<2x4xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[VAL_3]] : i32 to f32
// CHECK:             linalg.yield %[[SITOFP_0]] : f32
// CHECK:           } -> tensor<2x4xf32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[ARG0]] : f32) outs(%[[EMPTY_3]] : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_2]], %[[FILL_0]] : tensor<2x4xf32>, tensor<2x4xf32>) outs(%[[GENERIC_2]] : tensor<2x4xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f32):
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:             linalg.yield %[[ADDF_0]] : f32
// CHECK:           } -> tensor<2x4xf32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<4x2xf32>
// CHECK:           %[[TRANSPOSE_0:.*]] = linalg.transpose ins(%[[GENERIC_3]] : tensor<2x4xf32>) outs(%[[EMPTY_4]] : tensor<4x2xf32>) permutation = [1, 0]
// CHECK:           return %[[TRANSPOSE_0]] : tensor<4x2xf32>
// CHECK:         }
  tt.func @broadcast_transpose(%arg0: f32) -> tensor<4x2xf32> {
    %range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %exp = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %bcast = tt.broadcast %exp : tensor<1x4xi32> -> tensor<2x4xi32>
    %vals = arith.sitofp %bcast : tensor<2x4xi32> to tensor<2x4xf32>
    %splat = tt.splat %arg0 : f32 -> tensor<2x4xf32>
    %sum = arith.addf %vals, %splat : tensor<2x4xf32>
    %t = tt.trans %sum {order = array<i32: 1, 0>} : tensor<2x4xf32> -> tensor<4x2xf32>
    tt.return %t : tensor<4x2xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @reshape_collapse(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2x2xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[ARG0]] {{\[\[}}0, 1]] : tensor<2x2xf32> into tensor<4xf32>
// CHECK:           return %[[COLLAPSE_SHAPE_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @reshape_collapse(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
    %reshaped = tt.reshape %arg0 allow_reorder : tensor<2x2xf32> -> tensor<4xf32>
    tt.return %reshaped : tensor<4xf32>
  }
}

// -----

module {
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @bitcast_tensor(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_3]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<4xi32>) outs(%[[EMPTY_0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: f32):
// CHECK:             %[[BITCAST_0:.*]] = arith.bitcast %[[VAL_0]] : i32 to f32
// CHECK:             linalg.yield %[[BITCAST_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[GENERIC_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @bitcast_tensor(%arg0: tensor<4xi32>) -> tensor<4xf32> {
    %cast = tt.bitcast %arg0 : tensor<4xi32> -> tensor<4xf32>
    tt.return %cast : tensor<4xf32>
  }
}

// -----

module {
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @extern_unary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<4xf32>) outs(%[[ARG0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32):
// CHECK:             %[[SQRT_0:.*]] = math.sqrt %[[VAL_0]] : f32
// CHECK:             linalg.yield %[[SQRT_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[GENERIC_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @extern_unary(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__nv_sqrtf"} : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @cumsum_1d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_0]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[CUMSUM_0:.*]] = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>} ins(%[[FILL_0]] : tensor<4xf32>) outs(%[[EMPTY_1]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           return %[[CUMSUM_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @cumsum_1d() -> tensor<4xf32> {
    %c1 = arith.constant 1.0 : f32
    %input = tt.splat %c1 : f32 -> tensor<4xf32>
    %res = "tt.scan"(%input) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      tt.scan.return %sum : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @reduce_add(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_0]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[ALLOC_TENSOR_0:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           %[[INSERT_0:.*]] = tensor.insert %[[CONSTANT_0]] into %[[ALLOC_TENSOR_0]][] : tensor<f32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[FILL_0]] : tensor<4xf32>) outs(%[[INSERT_0]] : tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32) {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:               linalg.yield %[[ADDF_0]] : f32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<f32>
// CHECK:           return %[[EXTRACT_0]] : f32
// CHECK:         }
  tt.func @reduce_add() -> f32 {
    %c1 = arith.constant 1.0 : f32
    %input = tt.splat %c1 : f32 -> tensor<4xf32>
    %res = "tt.reduce"(%input) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %sum : f32
    }) {axis = 0 : i32} : (tensor<4xf32>) -> f32
    tt.return %res : f32
  }
}

// -----

module {
// CHECK-LABEL:   func.func @dot_2x2_2x2(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<2x2xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x2xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_0]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<2x2xf32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_1]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:           %[[MATMUL_0:.*]] = linalg.matmul ins(%[[FILL_0]], %[[FILL_0]] : tensor<2x2xf32>, tensor<2x2xf32>) outs(%[[FILL_1]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:           return %[[MATMUL_0]] : tensor<2x2xf32>
// CHECK:         }
  tt.func @dot_2x2_2x2() -> tensor<2x2xf32> {
    %c1 = arith.constant 1.0 : f32
    %a = tt.splat %c1 : f32 -> tensor<2x2xf32>
    %b = tt.splat %c1 : f32 -> tensor<2x2xf32>
    %c0 = arith.constant 0.0 : f32
    %c = tt.splat %c0 : f32 -> tensor<2x2xf32>
    %res = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<2x2xf32> * tensor<2x2xf32> -> tensor<2x2xf32>
    tt.return %res : tensor<2x2xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @split_join(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<2x2xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<2x2xf32> {
// CHECK:           %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [2, 1] [1, 1] : tensor<2x2xf32> to tensor<2xf32>
// CHECK:           %[[EXTRACT_SLICE_1:.*]] = tensor.extract_slice %[[ARG0]][0, 1] [2, 1] [1, 1] : tensor<2x2xf32> to tensor<2xf32>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x2xf32>
// CHECK:           %[[INSERT_SLICE_0:.*]] = tensor.insert_slice %[[EXTRACT_SLICE_0]] into %[[EMPTY_0]][0, 0] [2, 1] [1, 1] : tensor<2xf32> into tensor<2x2xf32>
// CHECK:           %[[INSERT_SLICE_1:.*]] = tensor.insert_slice %[[EXTRACT_SLICE_1]] into %[[INSERT_SLICE_0]][0, 1] [2, 1] [1, 1] : tensor<2xf32> into tensor<2x2xf32>
// CHECK:           return %[[INSERT_SLICE_1]] : tensor<2x2xf32>
// CHECK:         }
  tt.func @split_join(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %lhs, %rhs = tt.split %arg0 : tensor<2x2xf32> -> tensor<2xf32>
    %joined = tt.join %lhs, %rhs : tensor<2xf32> -> tensor<2x2xf32>
    tt.return %joined : tensor<2x2xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @cat_dim0(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<2x2xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x2xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_0]] : tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<1x2xf32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_1]] : tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<2x2xf32>
// CHECK:           %[[INSERT_SLICE_0:.*]] = tensor.insert_slice %[[FILL_0]] into %[[EMPTY_2]][0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
// CHECK:           %[[INSERT_SLICE_1:.*]] = tensor.insert_slice %[[FILL_1]] into %[[INSERT_SLICE_0]][1, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
// CHECK:           return %[[INSERT_SLICE_1]] : tensor<2x2xf32>
// CHECK:         }
  tt.func @cat_dim0() -> tensor<2x2xf32> {
    %c1 = arith.constant 1.0 : f32
    %c2 = arith.constant 2.0 : f32
    %a = tt.splat %c1 : f32 -> tensor<1x2xf32>
    %b = tt.splat %c2 : f32 -> tensor<1x2xf32>
    %cat = tt.cat %a, %b : tensor<1x2xf32> -> tensor<2x2xf32>
    tt.return %cat : tensor<2x2xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @assert_scalar(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[ARG0]], %[[CONSTANT_0]] : i32
// CHECK:           cf.assert %[[CMPI_0]], "Assertion `x > 0` failed"
// CHECK:           return
// CHECK:         }
  tt.func @assert_scalar(%arg0: i32) {
    %zero = arith.constant 0 : i32
    %cond = arith.cmpi sgt, %arg0, %zero : i32
    tt.assert %cond, "x > 0" : i1
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   func.func @dense_constant(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<2xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_0]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return %[[FILL_0]] : tensor<2xf32>
// CHECK:         }
  tt.func @dense_constant() -> tensor<2xf32> {
    %cst = arith.constant dense<1.0> : tensor<2xf32>
    tt.return %cst : tensor<2xf32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @call_callee(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           return %[[ARG0]] : tensor<4xf32>
// CHECK:         }
  tt.func @call_callee(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    tt.return %arg0 : tensor<4xf32>
  }

// CHECK-LABEL:   func.func @call_caller(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[VAL_0:.*]] = call @call_callee(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]], %[[ARG6]]) : (tensor<4xf32>, i32, i32, i32, i32, i32, i32) -> tensor<4xf32>
// CHECK:           return %[[VAL_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @call_caller(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.call @call_callee(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// -----

module {
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @fptofp_rtne(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf16> {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf16>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<4xf32>) outs(%[[EMPTY_0]] : tensor<4xf16>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f16):
// CHECK:             %[[TRUNCF_0:.*]] = arith.truncf %[[VAL_0]] : f32 to f16
// CHECK:             linalg.yield %[[TRUNCF_0]] : f16
// CHECK:           } -> tensor<4xf16>
// CHECK:           return %[[GENERIC_0]] : tensor<4xf16>
// CHECK:         }
  tt.func @fptofp_rtne(%arg0: tensor<4xf32>) -> tensor<4xf16> {
    %res = tt.fp_to_fp %arg0, rounding = rtne : tensor<4xf32> -> tensor<4xf16>
    tt.return %res : tensor<4xf16>
  }
}

// -----

module {
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @clamp_all(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_6]], #[[$ATTR_6]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<4xf32>, tensor<4xf32>) outs(%[[ARG0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32):
// CHECK:             %[[MAXIMUMF_0:.*]] = arith.maximumf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:             linalg.yield %[[MAXIMUMF_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_6]], #[[$ATTR_6]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[ARG2]] : tensor<4xf32>, tensor<4xf32>) outs(%[[GENERIC_0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32):
// CHECK:             %[[MINIMUMF_0:.*]] = arith.minimumf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             linalg.yield %[[MINIMUMF_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[GENERIC_1]] : tensor<4xf32>
// CHECK:         }
  tt.func @clamp_all(%x: tensor<4xf32>, %min: tensor<4xf32>,
                     %max: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.clampf %x, %min, %max, propagateNan = all : tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// -----

module {
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @precise_math(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_7]], #[[$ATTR_7]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<4xf32>) outs(%[[ARG0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32):
// CHECK:             %[[SQRT_0:.*]] = math.sqrt %[[VAL_0]] : f32
// CHECK:             linalg.yield %[[SQRT_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_7]], #[[$ATTR_7]], #[[$ATTR_7]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[ARG1]] : tensor<4xf32>, tensor<4xf32>) outs(%[[GENERIC_0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32):
// CHECK:             %[[DIVF_0:.*]] = arith.divf %[[VAL_2]], %[[VAL_3]] : f32
// CHECK:             linalg.yield %[[DIVF_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[GENERIC_1]] : tensor<4xf32>
// CHECK:         }
  tt.func @precise_math(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %sqrt = tt.precise_sqrt %arg0 : tensor<4xf32>
    %div = tt.precise_divf %sqrt, %arg1 : tensor<4xf32>
    tt.return %div : tensor<4xf32>
  }
}

// -----

module {
// CHECK: #[[$ATTR_8:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @mulhiui(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xi32> {
// CHECK:           %[[GENERIC_0:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_8]], #[[$ATTR_8]], #[[$ATTR_8]], #[[$ATTR_8]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<4xi32>, tensor<4xi32>) outs(%[[ARG0]], %[[ARG0]] : tensor<4xi32>, tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[VAL_4:.*]], %[[MULUI_EXTENDED_0:.*]] = arith.mului_extended %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:             linalg.yield %[[VAL_4]], %[[MULUI_EXTENDED_0]] : i32, i32
// CHECK:           } -> (tensor<4xi32>, tensor<4xi32>)
// CHECK:           return %[[VAL_5:.*]]#1 : tensor<4xi32>
// CHECK:         }
  tt.func @mulhiui(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %res = tt.mulhiui %arg0, %arg1 : tensor<4xi32>
    tt.return %res : tensor<4xi32>
  }
}

// -----

module {
// CHECK-LABEL:   func.func @unsplat_scalar(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<1xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[CONSTANT_0]]] : tensor<1xf32>
// CHECK:           return %[[EXTRACT_0]] : f32
// CHECK:         }
  tt.func @unsplat_scalar(%arg0: tensor<1xf32>) -> f32 {
    %res = tt.unsplat %arg0 : tensor<1xf32>
    tt.return %res : f32
  }
}

// -----

module {
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @argmax_4(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : f32) outs(%[[EMPTY_0]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_9]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_2]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_3]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4xf32>, tensor<4xi32>) outs(%[[FILL_1]], %[[FILL_2]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: i32) {
// CHECK:               %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_1]], %[[VAL_3]] : f32
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_0]], %[[CMPI_0]] : i1
// CHECK:               %[[CMPF_1:.*]] = arith.cmpf ogt, %[[VAL_1]], %[[VAL_3]] : f32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPF_1]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_1]], %[[VAL_3]] : f32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : f32, i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]]#1[] : tensor<i32>
// CHECK:           return %[[EXTRACT_0]] : i32
// CHECK:         }
  tt.func @argmax_4() -> i32 {
    %vals = arith.constant dense<0.0> : tensor<4xf32>
    %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %res:2 = "tt.reduce"(%vals, %idx) <{axis = 0 : i32}> ({
    ^bb0(%v: f32, %i: i32, %v_acc: f32, %i_acc: i32):
      %eq = arith.cmpf oeq, %v, %v_acc : f32
      %lt = arith.cmpi slt, %i, %i_acc : i32
      %tie = arith.andi %eq, %lt : i1
      %gt = arith.cmpf ogt, %v, %v_acc : f32
      %pick = arith.ori %gt, %tie : i1
      %v_out = arith.select %pick, %v, %v_acc : f32
      %i_out = arith.select %pick, %i, %i_acc : i32
      tt.reduce.return %v_out, %i_out : f32, i32
    }) : (tensor<4xf32>, tensor<4xi32>) -> (f32, i32)
    tt.return %res#1 : i32
  }
}

// -----

module {
// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @argmin_4(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0x7F800000 : f32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : f32) outs(%[[EMPTY_0]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_10]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_2]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_3]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4xf32>, tensor<4xi32>) outs(%[[FILL_1]], %[[FILL_2]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: i32) {
// CHECK:               %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_1]], %[[VAL_3]] : f32
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_0]], %[[CMPI_0]] : i1
// CHECK:               %[[CMPF_1:.*]] = arith.cmpf olt, %[[VAL_1]], %[[VAL_3]] : f32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPF_1]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_1]], %[[VAL_3]] : f32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : f32, i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]]#1[] : tensor<i32>
// CHECK:           return %[[EXTRACT_0]] : i32
// CHECK:         }
  tt.func @argmin_4() -> i32 {
    %vals = arith.constant dense<0.0> : tensor<4xf32>
    %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %res:2 = "tt.reduce"(%vals, %idx) <{axis = 0 : i32}> ({
    ^bb0(%v: f32, %i: i32, %v_acc: f32, %i_acc: i32):
      %eq = arith.cmpf oeq, %v, %v_acc : f32
      %lt = arith.cmpi slt, %i, %i_acc : i32
      %tie = arith.andi %eq, %lt : i1
      %ltv = arith.cmpf olt, %v, %v_acc : f32
      %pick = arith.ori %ltv, %tie : i1
      %v_out = arith.select %pick, %v, %v_acc : f32
      %i_out = arith.select %pick, %i, %i_acc : i32
      tt.reduce.return %v_out, %i_out : f32, i32
    }) : (tensor<4xf32>, tensor<4xi32>) -> (f32, i32)
    tt.return %res#1 : i32
  }
}

// -----

module {
// CHECK: #[[$ATTR_11:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @minmax_select(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_11]], #[[$ATTR_11]], #[[$ATTR_11]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<4xf32>, tensor<4xf32>) outs(%[[ARG0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32):
// CHECK:             %[[MAXIMUMF_0:.*]] = arith.maximumf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:             linalg.yield %[[MAXIMUMF_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[GENERIC_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @minmax_select(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %cmp = arith.cmpf ogt, %arg0, %arg1 : tensor<4xf32>
    %sel = arith.select %cmp, %arg0, %arg1 : tensor<4xi1>, tensor<4xf32>
    tt.return %sel : tensor<4xf32>
  }
}

// -----

module {
// CHECK: #[[$ATTR_12:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @extern_binary(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xf32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> tensor<4xf32> {
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_12]], #[[$ATTR_12]], #[[$ATTR_12]]], iterator_types = ["parallel"]} ins(%[[ARG0]], %[[ARG1]] : tensor<4xf32>, tensor<4xf32>) outs(%[[ARG0]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32):
// CHECK:             %[[VAL_3:.*]] = math.atan2 %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:             linalg.yield %[[VAL_3]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           return %[[GENERIC_0]] : tensor<4xf32>
// CHECK:         }
  tt.func @extern_binary(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %res = tt.extern_elementwise %arg0, %arg1 {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}
