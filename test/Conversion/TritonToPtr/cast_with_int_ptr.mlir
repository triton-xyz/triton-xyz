// RUN: triton-shared-opt --split-input-file --triton-tensor-ptr-to-linalg --triton-to-ptr %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func public @cast_with_int_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {noinline = false} {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset i32 : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG2]], %[[TYPE_OFFSET_0]] : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[UNREALIZED_CONVERSION_CAST_1]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_1:.*]] = ptr.type_offset i8 : i32
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[CONSTANT_0]], %[[TYPE_OFFSET_1]] : i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[UNREALIZED_CONVERSION_CAST_0]] : !ptr.ptr<#ptr.generic_space> to i64
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[UNREALIZED_CONVERSION_CAST_2]] : i64 to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[TYPE_OFFSET_2:.*]] = ptr.type_offset i32 : i64
// CHECK:           %[[MULI_2:.*]] = arith.muli %[[UNREALIZED_CONVERSION_CAST_2]], %[[TYPE_OFFSET_2]] : i64
// CHECK:           %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[UNREALIZED_CONVERSION_CAST_3]], %[[MULI_2]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           %[[LOAD_0:.*]] = ptr.load %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space> -> i32
// CHECK:           ptr.store %[[LOAD_0]], %[[PTR_ADD_2]] : i32, !ptr.ptr<#ptr.generic_space>
// CHECK:           tt.return
// CHECK:         }
  tt.func public @cast_with_int_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %idx: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg0, %idx : !tt.ptr<i32>, i32
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<i8>
    %2 = tt.addptr %1, %c1_i32 : !tt.ptr<i8>, i32
    %3 = tt.bitcast %2 : !tt.ptr<i8> -> !tt.ptr<i32>
    %4 = tt.ptr_to_int %arg1 : !tt.ptr<i32> -> i64
    %5 = tt.int_to_ptr %4 : i64 -> !tt.ptr<i32>
    %6 = tt.addptr %5, %4 : !tt.ptr<i32>, i64
    %7 = tt.load %3 : !tt.ptr<i32>
    tt.store %6, %7 : !tt.ptr<i32>
    tt.return
  }
}
