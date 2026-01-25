// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file --verify-diagnostics %s

module {
  tt.func @cumsum_axis0_rank2() -> tensor<2x2xf32> {
    %c1 = arith.constant 1.0 : f32
    %input = tt.splat %c1 : f32 -> tensor<2x2xf32>
    // expected-error@+1 {{CumSum computation only supports axis == rank - 1}}
    %res = "tt.scan"(%input) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      tt.scan.return %sum : f32
    }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    tt.return %res : tensor<2x2xf32>
  }
}

// -----

module {
  tt.func @cumsum_rank3() -> tensor<2x2x2xf32> {
    %c1 = arith.constant 1.0 : f32
    %input = tt.splat %c1 : f32 -> tensor<2x2x2xf32>
    // expected-error@+1 {{CumSum op only takes tensors of rank 1 & 2.}}
    %res = "tt.scan"(%input) <{axis = 2 : i32, reverse = false}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      tt.scan.return %sum : f32
    }) : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    tt.return %res : tensor<2x2x2xf32>
  }
}
