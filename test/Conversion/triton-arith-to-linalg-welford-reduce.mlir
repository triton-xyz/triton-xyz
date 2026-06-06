// RUN: triton-xyz-opt --split-input-file --triton-arith-to-linalg %s | FileCheck %s

module {
// CHECK-LABEL: func.func @welford_reduce_vector() -> f32
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[INIT0:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}} : tensor<f32>) -> tensor<f32>
// CHECK:       %[[INIT1:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}} : tensor<f32>) -> tensor<f32>
// CHECK:       %[[INIT2:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}} : tensor<f32>) -> tensor<f32>
// CHECK:       %[[REDUCE:.*]]:3 = linalg.reduce
// CHECK-SAME:  outs(%[[INIT0]], %[[INIT1]], %[[INIT2]] : tensor<f32>, tensor<f32>, tensor<f32>) dimensions = [0]
// CHECK:       arith.maxnumf
// CHECK:       arith.subf
// CHECK:       linalg.yield {{.*}} : f32, f32, f32
// CHECK:       %[[EXTRACT:.*]] = tensor.extract %[[REDUCE]]#2[] : tensor<f32>
// CHECK:       return %[[EXTRACT]] : f32
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
// CHECK-LABEL: func.func @welford_reduce_rows() -> tensor<2xf32>
// CHECK:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[INIT0:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}} : tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[INIT1:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}} : tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[INIT2:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}} : tensor<2xf32>) -> tensor<2xf32>
// CHECK:       %[[REDUCE:.*]]:3 = linalg.reduce
// CHECK-SAME:  outs(%[[INIT0]], %[[INIT1]], %[[INIT2]] : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) dimensions = [1]
// CHECK:       arith.maxnumf
// CHECK:       arith.subf
// CHECK:       linalg.yield {{.*}} : f32, f32, f32
// CHECK:       return %[[REDUCE]]#2 : tensor<2xf32>
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
