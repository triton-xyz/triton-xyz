// RUN: triton-xyz-opt --split-input-file --triton-arith-to-linalg %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @welford_reduce_vector() -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_0]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_1]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_2]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_3:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_3]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_4:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_4]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[REDUCE_0:.*]]:3 = linalg.reduce ins(%[[FILL_0]], %[[FILL_0]], %[[FILL_1]] : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%[[FILL_2]], %[[FILL_3]], %[[FILL_4]] : tensor<f32>, tensor<f32>, tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32) {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[VAL_4]] : f32
// CHECK:               %[[MAXNUMF_0:.*]] = arith.maxnumf %[[ADDF_0]], %[[CONSTANT_1]] : f32
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:               %[[ADDF_1:.*]] = arith.addf %[[MULF_0]], %[[MULF_1]] : f32
// CHECK:               %[[DIVF_0:.*]] = arith.divf %[[ADDF_1]], %[[MAXNUMF_0]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[MULF_0]], %[[VAL_0]] : f32
// CHECK:               %[[ADDF_2:.*]] = arith.addf %[[VAL_2]], %[[MULF_2]] : f32
// CHECK:               %[[ADDF_3:.*]] = arith.addf %[[ADDF_2]], %[[VAL_5]] : f32
// CHECK:               %[[MULF_3:.*]] = arith.mulf %[[MULF_1]], %[[VAL_3]] : f32
// CHECK:               %[[ADDF_4:.*]] = arith.addf %[[ADDF_3]], %[[MULF_3]] : f32
// CHECK:               %[[MULF_4:.*]] = arith.mulf %[[ADDF_0]], %[[DIVF_0]] : f32
// CHECK:               %[[MULF_5:.*]] = arith.mulf %[[MULF_4]], %[[DIVF_0]] : f32
// CHECK:               %[[SUBF_0:.*]] = arith.subf %[[ADDF_4]], %[[MULF_5]] : f32
// CHECK:               linalg.yield %[[DIVF_0]], %[[ADDF_0]], %[[SUBF_0]] : f32, f32, f32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]]#2[] : tensor<f32>
// CHECK:           return %[[EXTRACT_0]] : f32
// CHECK:         }
  tt.func @welford_reduce_vector() -> f32 {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %mean = tt.splat %c1 : f32 -> tensor<4xf32>
    %count = tt.splat %c1 : f32 -> tensor<4xf32>
    %m2 = tt.splat %c0 : f32 -> tensor<4xf32>
    %res:3 = "tt.reduce"(%mean, %count, %m2) <{axis = 0 : i32}> ({
    ^bb0(%mean_x: f32, %count_x: f32, %m2_x: f32, %mean_y: f32, %count_y: f32, %m2_y: f32):
      %count_out = arith.addf %count_x, %count_y : f32
      %safe_count = arith.maxnumf %count_out, %c1 : f32
      %mean_count_x = arith.mulf %mean_x, %count_x : f32
      %mean_count_y = arith.mulf %mean_y, %count_y : f32
      %mean_sum = arith.addf %mean_count_x, %mean_count_y : f32
      %mean_out = arith.divf %mean_sum, %safe_count : f32
      %m2_term_x = arith.mulf %mean_count_x, %mean_x : f32
      %m2_acc_x = arith.addf %m2_x, %m2_term_x : f32
      %m2_acc_y = arith.addf %m2_acc_x, %m2_y : f32
      %m2_term_y = arith.mulf %mean_count_y, %mean_y : f32
      %m2_sum = arith.addf %m2_acc_y, %m2_term_y : f32
      %mean_scaled = arith.mulf %count_out, %mean_out : f32
      %mean_scaled_sq = arith.mulf %mean_scaled, %mean_out : f32
      %m2_out = arith.subf %m2_sum, %mean_scaled_sq : f32
      tt.reduce.return %mean_out, %count_out, %m2_out : f32, f32, f32
    }) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (f32, f32, f32)
    tt.return %res#2 : f32
  }
}

// -----

module {
// CHECK-LABEL:   func.func @welford_reduce_rows() -> tensor<2xf32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_0]] : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_1]] : tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_2]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[FILL_3:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_3]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<2xf32>
// CHECK:           %[[FILL_4:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_4]] : tensor<2xf32>) -> tensor<2xf32>
// CHECK:           %[[REDUCE_0:.*]]:3 = linalg.reduce ins(%[[FILL_0]], %[[FILL_0]], %[[FILL_1]] : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) outs(%[[FILL_2]], %[[FILL_3]], %[[FILL_4]] : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) dimensions = [1]
// CHECK:             (%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32, %[[VAL_2:.*]]: f32, %[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32) {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[VAL_1]], %[[VAL_4]] : f32
// CHECK:               %[[MAXNUMF_0:.*]] = arith.maxnumf %[[ADDF_0]], %[[CONSTANT_1]] : f32
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:               %[[ADDF_1:.*]] = arith.addf %[[MULF_0]], %[[MULF_1]] : f32
// CHECK:               %[[DIVF_0:.*]] = arith.divf %[[ADDF_1]], %[[MAXNUMF_0]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[MULF_0]], %[[VAL_0]] : f32
// CHECK:               %[[ADDF_2:.*]] = arith.addf %[[VAL_2]], %[[MULF_2]] : f32
// CHECK:               %[[ADDF_3:.*]] = arith.addf %[[ADDF_2]], %[[VAL_5]] : f32
// CHECK:               %[[MULF_3:.*]] = arith.mulf %[[MULF_1]], %[[VAL_3]] : f32
// CHECK:               %[[ADDF_4:.*]] = arith.addf %[[ADDF_3]], %[[MULF_3]] : f32
// CHECK:               %[[MULF_4:.*]] = arith.mulf %[[ADDF_0]], %[[DIVF_0]] : f32
// CHECK:               %[[MULF_5:.*]] = arith.mulf %[[MULF_4]], %[[DIVF_0]] : f32
// CHECK:               %[[SUBF_0:.*]] = arith.subf %[[ADDF_4]], %[[MULF_5]] : f32
// CHECK:               linalg.yield %[[DIVF_0]], %[[ADDF_0]], %[[SUBF_0]] : f32, f32, f32
// CHECK:             }
// CHECK:           return %[[REDUCE_0]]#2 : tensor<2xf32>
// CHECK:         }
  tt.func @welford_reduce_rows() -> tensor<2xf32> {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %mean = tt.splat %c1 : f32 -> tensor<2x4xf32>
    %count = tt.splat %c1 : f32 -> tensor<2x4xf32>
    %m2 = tt.splat %c0 : f32 -> tensor<2x4xf32>
    %res:3 = "tt.reduce"(%mean, %count, %m2) <{axis = 1 : i32}> ({
    ^bb0(%mean_x: f32, %count_x: f32, %m2_x: f32, %mean_y: f32, %count_y: f32, %m2_y: f32):
      %count_out = arith.addf %count_x, %count_y : f32
      %safe_count = arith.maxnumf %count_out, %c1 : f32
      %mean_count_x = arith.mulf %mean_x, %count_x : f32
      %mean_count_y = arith.mulf %mean_y, %count_y : f32
      %mean_sum = arith.addf %mean_count_x, %mean_count_y : f32
      %mean_out = arith.divf %mean_sum, %safe_count : f32
      %m2_term_x = arith.mulf %mean_count_x, %mean_x : f32
      %m2_acc_x = arith.addf %m2_x, %m2_term_x : f32
      %m2_acc_y = arith.addf %m2_acc_x, %m2_y : f32
      %m2_term_y = arith.mulf %mean_count_y, %mean_y : f32
      %m2_sum = arith.addf %m2_acc_y, %m2_term_y : f32
      %mean_scaled = arith.mulf %count_out, %mean_out : f32
      %mean_scaled_sq = arith.mulf %mean_scaled, %mean_out : f32
      %m2_out = arith.subf %m2_sum, %mean_scaled_sq : f32
      tt.reduce.return %mean_out, %count_out, %m2_out : f32, f32, f32
    }) : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
    tt.return %res#2 : tensor<2xf32>
  }
}
