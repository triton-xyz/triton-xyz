// RUN: triton-xyz-opt --split-input-file --triton-pids-to-func-args %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @program_info(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i32 {
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG3]], %[[ARG0]] : i32
// CHECK:           tt.return %[[ADDI_0]] : i32
// CHECK:         }
  tt.func @program_info() -> i32 {
    %pid = tt.get_program_id x : i32
    %nprog = tt.get_num_programs x : i32
    %sum = arith.addi %pid, %nprog : i32
    tt.return %sum : i32
  }
}
