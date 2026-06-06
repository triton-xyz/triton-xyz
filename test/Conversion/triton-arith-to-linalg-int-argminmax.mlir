// RUN: triton-xyz-opt --split-input-file --triton-arith-to-linalg %s | FileCheck %s

module {
// CHECK-LABEL: func.func @argmax_i32() -> i32
// CHECK:       %[[MIN:.*]] = arith.constant -2147483648 : i32
// CHECK:       %[[VALUE_INIT:.*]] = linalg.fill ins(%[[MIN]] : i32)
// CHECK:       %[[REDUCE:.*]]:2 = linalg.reduce
// CHECK-SAME:  outs(%[[VALUE_INIT]]
// CHECK-SAME:  dimensions = [0]
// CHECK:       (%[[VALUE:.*]]: i32, %[[INDEX:.*]]: i32, %[[ACC_VALUE:.*]]: i32, %[[ACC_INDEX:.*]]: i32) {
// CHECK:         %[[EQ:.*]] = arith.cmpi eq, %[[VALUE]], %[[ACC_VALUE]] : i32
// CHECK:         %[[EARLIER:.*]] = arith.cmpi slt, %[[INDEX]], %[[ACC_INDEX]] : i32
// CHECK:         %[[TIE:.*]] = arith.andi %[[EQ]], %[[EARLIER]] : i1
// CHECK:         %[[BETTER:.*]] = arith.cmpi sgt, %[[VALUE]], %[[ACC_VALUE]] : i32
// CHECK:         %[[PICK:.*]] = arith.ori %[[BETTER]], %[[TIE]] : i1
// CHECK:         arith.select %[[PICK]], %[[VALUE]], %[[ACC_VALUE]] : i32
// CHECK:         arith.select %[[PICK]], %[[INDEX]], %[[ACC_INDEX]] : i32
// CHECK:         linalg.yield
// CHECK:       %[[INDEX_RESULT:.*]] = tensor.extract %[[REDUCE]]#1[] : tensor<i32>
// CHECK:       return %[[INDEX_RESULT]] : i32
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
// CHECK-LABEL: func.func @argmin_i32() -> i32
// CHECK:       %[[MAX:.*]] = arith.constant 2147483647 : i32
// CHECK:       %[[VALUE_INIT:.*]] = linalg.fill ins(%[[MAX]] : i32)
// CHECK:       %[[REDUCE:.*]]:2 = linalg.reduce
// CHECK-SAME:  outs(%[[VALUE_INIT]]
// CHECK-SAME:  dimensions = [0]
// CHECK:       (%[[VALUE:.*]]: i32, %[[INDEX:.*]]: i32, %[[ACC_VALUE:.*]]: i32, %[[ACC_INDEX:.*]]: i32) {
// CHECK:         %[[EQ:.*]] = arith.cmpi eq, %[[VALUE]], %[[ACC_VALUE]] : i32
// CHECK:         %[[EARLIER:.*]] = arith.cmpi slt, %[[INDEX]], %[[ACC_INDEX]] : i32
// CHECK:         %[[TIE:.*]] = arith.andi %[[EQ]], %[[EARLIER]] : i1
// CHECK:         %[[BETTER:.*]] = arith.cmpi slt, %[[VALUE]], %[[ACC_VALUE]] : i32
// CHECK:         %[[PICK:.*]] = arith.ori %[[BETTER]], %[[TIE]] : i1
// CHECK:         arith.select %[[PICK]], %[[VALUE]], %[[ACC_VALUE]] : i32
// CHECK:         arith.select %[[PICK]], %[[INDEX]], %[[ACC_INDEX]] : i32
// CHECK:         linalg.yield
// CHECK:       %[[INDEX_RESULT:.*]] = tensor.extract %[[REDUCE]]#1[] : tensor<i32>
// CHECK:       return %[[INDEX_RESULT]] : i32
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
