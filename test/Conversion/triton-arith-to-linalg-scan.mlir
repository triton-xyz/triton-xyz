// RUN: triton-xyz-opt --split-input-file --triton-arith-to-linalg %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @scan_max_1d
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4xf32>
  // CHECK: %[[FIRST:.+]] = tensor.extract %arg0[%[[C0]]] : tensor<4xf32>
  // CHECK: %[[SEEDED:.+]] = tensor.insert %[[FIRST]] into %[[EMPTY]][%[[C0]]] : tensor<4xf32>
  // CHECK: %[[FOR:.+]]:2 = scf.for %[[IV:.+]] = %[[C1]] to %[[C4]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[FIRST]], %[[OUT:.+]] = %[[SEEDED]]) -> (f32, tensor<4xf32>) {
  // CHECK:   %[[CUR:.+]] = tensor.extract %arg0[%[[IV]]] : tensor<4xf32>
  // CHECK:   %[[MAX:.+]] = arith.maxnumf %[[ACC]], %[[CUR]] : f32
  // CHECK:   %[[NEXT:.+]] = tensor.insert %[[MAX]] into %[[OUT]][%[[IV]]] : tensor<4xf32>
  // CHECK:   scf.yield %[[MAX]], %[[NEXT]] : f32, tensor<4xf32>
  // CHECK: }
  // CHECK: return %[[FOR]]#1 : tensor<4xf32>
  tt.func @scan_max_1d(%input: tensor<4xf32>) -> tensor<4xf32> {
    %res = "tt.scan"(%input) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %max = arith.maxnumf %arg0, %arg1 : f32
      tt.scan.return %max : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @scan_max_1d_reverse
  // CHECK: %[[C3:.+]] = arith.constant 3 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4xf32>
  // CHECK: %[[FIRST:.+]] = tensor.extract %arg0[%[[C3]]] : tensor<4xf32>
  // CHECK: %[[SEEDED:.+]] = tensor.insert %[[FIRST]] into %[[EMPTY]][%[[C3]]] : tensor<4xf32>
  // CHECK: %[[FOR:.+]]:2 = scf.for %[[IV:.+]] = %[[C1]] to %[[C4]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[FIRST]], %[[OUT:.+]] = %[[SEEDED]]) -> (f32, tensor<4xf32>) {
  // CHECK:   %[[POS:.+]] = arith.subi %[[C3]], %[[IV]] : index
  // CHECK:   %[[CUR:.+]] = tensor.extract %arg0[%[[POS]]] : tensor<4xf32>
  // CHECK:   %[[MAX:.+]] = arith.maxnumf %[[ACC]], %[[CUR]] : f32
  // CHECK:   %[[NEXT:.+]] = tensor.insert %[[MAX]] into %[[OUT]][%[[POS]]] : tensor<4xf32>
  // CHECK:   scf.yield %[[MAX]], %[[NEXT]] : f32, tensor<4xf32>
  // CHECK: }
  // CHECK: return %[[FOR]]#1 : tensor<4xf32>
  tt.func @scan_max_1d_reverse(%input: tensor<4xf32>) -> tensor<4xf32> {
    %res = "tt.scan"(%input) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %max = arith.maxnumf %arg0, %arg1 : f32
      tt.scan.return %max : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    tt.return %res : tensor<4xf32>
  }
}
