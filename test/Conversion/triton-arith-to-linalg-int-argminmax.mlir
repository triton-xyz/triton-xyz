// RUN: triton-xyz-opt --split-input-file --triton-arith-to-linalg %s | FileCheck %s

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @argmax_i32() -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant -2147483648 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_2]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_3]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4xi32>, tensor<4xi32>) outs(%[[FILL_1]], %[[FILL_2]] : tensor<i32>, tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:               %[[CMPI_1:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:               %[[CMPI_2:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPI_2]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : i32, i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]]#1[] : tensor<i32>
// CHECK:           return %[[EXTRACT_0]] : i32
// CHECK:         }
  tt.func @argmax_i32() -> i32 {
    %vals = arith.constant dense<0> : tensor<4xi32>
    %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %res:2 = "tt.reduce"(%vals, %idx) <{axis = 0 : i32}> ({
    ^bb0(%v: i32, %i: i32, %v_acc: i32, %i_acc: i32):
      %eq = arith.cmpi eq, %v, %v_acc : i32
      %lt = arith.cmpi slt, %i, %i_acc : i32
      %tie = arith.andi %eq, %lt : i1
      %gt = arith.cmpi sgt, %v, %v_acc : i32
      %pick = arith.ori %gt, %tie : i1
      %v_out = arith.select %pick, %v, %v_acc : i32
      %i_out = arith.select %pick, %i, %i_acc : i32
      tt.reduce.return %v_out, %i_out : i32, i32
    }) : (tensor<4xi32>, tensor<4xi32>) -> (i32, i32)
    tt.return %res#1 : i32
  }
}

// -----

module {
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @argmin_i32() -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2147483647 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_0]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_2]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_3]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4xi32>, tensor<4xi32>) outs(%[[FILL_1]], %[[FILL_2]] : tensor<i32>, tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:               %[[CMPI_1:.*]] = arith.cmpi slt, %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:               %[[CMPI_2:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPI_2]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : i32, i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]]#1[] : tensor<i32>
// CHECK:           return %[[EXTRACT_0]] : i32
// CHECK:         }
  tt.func @argmin_i32() -> i32 {
    %vals = arith.constant dense<0> : tensor<4xi32>
    %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %res:2 = "tt.reduce"(%vals, %idx) <{axis = 0 : i32}> ({
    ^bb0(%v: i32, %i: i32, %v_acc: i32, %i_acc: i32):
      %eq = arith.cmpi eq, %v, %v_acc : i32
      %lt = arith.cmpi slt, %i, %i_acc : i32
      %tie = arith.andi %eq, %lt : i1
      %ltv = arith.cmpi slt, %v, %v_acc : i32
      %pick = arith.ori %ltv, %tie : i1
      %v_out = arith.select %pick, %v, %v_acc : i32
      %i_out = arith.select %pick, %i, %i_acc : i32
      tt.reduce.return %v_out, %i_out : i32, i32
    }) : (tensor<4xi32>, tensor<4xi32>) -> (i32, i32)
    tt.return %res#1 : i32
  }
}
