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
  // CHECK-LABEL: func.func @scan_multi_input_1d
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[MIN_I64:.+]] = arith.constant -2147483648 : i64
  // CHECK: %[[MAX_I64:.+]] = arith.constant 2147483647 : i64
  // CHECK: %[[EMPTY_F:.+]] = tensor.empty() : tensor<4xf32>
  // CHECK: %[[EMPTY_I:.+]] = tensor.empty() : tensor<4xi32>
  // CHECK: %[[FIRST_F:.+]] = tensor.extract %arg0[%[[C0]]] : tensor<4xf32>
  // CHECK: %[[SEEDED_F:.+]] = tensor.insert %[[FIRST_F]] into %[[EMPTY_F]][%[[C0]]] : tensor<4xf32>
  // CHECK: %[[FIRST_I:.+]] = tensor.extract %arg1[%[[C0]]] : tensor<4xi32>
  // CHECK: %[[SEEDED_I:.+]] = tensor.insert %[[FIRST_I]] into %[[EMPTY_I]][%[[C0]]] : tensor<4xi32>
  // CHECK: %[[FOR:.+]]:4 = scf.for %[[IV:.+]] = %[[C1]] to %[[C4]] step %[[C1]] iter_args(%[[ACC_F:.+]] = %[[FIRST_F]], %[[ACC_I:.+]] = %[[FIRST_I]], %[[OUT_F:.+]] = %[[SEEDED_F]], %[[OUT_I:.+]] = %[[SEEDED_I]]) -> (f32, i32, tensor<4xf32>, tensor<4xi32>) {
  // CHECK:   %[[CUR_F:.+]] = tensor.extract %arg0[%[[IV]]] : tensor<4xf32>
  // CHECK:   %[[CUR_I:.+]] = tensor.extract %arg1[%[[IV]]] : tensor<4xi32>
  // CHECK:   %[[SUM_F:.+]] = arith.addf %[[ACC_F]], %[[CUR_F]] : f32
  // CHECK:   %[[ACC_I64:.+]] = arith.extsi %[[ACC_I]] : i32 to i64
  // CHECK:   %[[CUR_I64:.+]] = arith.extsi %[[CUR_I]] : i32 to i64
  // CHECK:   %[[SUM_I64:.+]] = arith.addi %[[ACC_I64]], %[[CUR_I64]] : i64
  // CHECK:   %[[LE:.+]] = arith.cmpi sle, %[[SUM_I64]], %[[MAX_I64]] : i64
  // CHECK:   %[[GE:.+]] = arith.cmpi sge, %[[SUM_I64]], %[[MIN_I64]] : i64
  // CHECK:   %[[IN_BOUNDS:.+]] = arith.andi %[[LE]], %[[GE]] : i1
  // CHECK:   cf.assert %[[IN_BOUNDS]], "Assertion `int32 overflow detected for operation add` failed"
  // CHECK:   %[[SUM_I:.+]] = arith.addi %[[ACC_I]], %[[CUR_I]] : i32
  // CHECK:   %[[NEXT_F:.+]] = tensor.insert %[[SUM_F]] into %[[OUT_F]][%[[IV]]] : tensor<4xf32>
  // CHECK:   %[[NEXT_I:.+]] = tensor.insert %[[SUM_I]] into %[[OUT_I]][%[[IV]]] : tensor<4xi32>
  // CHECK:   scf.yield %[[SUM_F]], %[[SUM_I]], %[[NEXT_F]], %[[NEXT_I]] : f32, i32, tensor<4xf32>, tensor<4xi32>
  // CHECK: }
  // CHECK: return %[[FOR]]#2, %[[FOR]]#3 : tensor<4xf32>, tensor<4xi32>
  tt.func @scan_multi_input_1d(%vals: tensor<4xf32>, %idxs: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %res:2 = "tt.scan"(%vals, %idxs) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a_val: f32, %a_idx: i32, %b_val: f32, %b_idx: i32):
      %sum = arith.addf %a_val, %b_val : f32
      %a_idx64 = arith.extsi %a_idx : i32 to i64
      %b_idx64 = arith.extsi %b_idx : i32 to i64
      %sum64 = arith.addi %a_idx64, %b_idx64 : i64
      %le = arith.cmpi sle, %sum64, %c2147483647_i64 : i64
      %ge = arith.cmpi sge, %sum64, %c-2147483648_i64 : i64
      %in = arith.andi %le, %ge : i1
      tt.assert %in, "int32 overflow detected for operation add" : i1
      %sum_idx = arith.addi %a_idx, %b_idx : i32
      tt.scan.return %sum, %sum_idx : f32, i32
    }) : (tensor<4xf32>, tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    tt.return %res#0, %res#1 : tensor<4xf32>, tensor<4xi32>
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
