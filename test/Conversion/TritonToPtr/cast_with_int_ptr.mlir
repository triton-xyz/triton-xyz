// RUN: triton-shared-opt --split-input-file --triton-arith-to-linalg="tensor-ptr-to-linalg" --triton-to-ptr %s | FileCheck %s

module {
// CHECK-LABEL:   func.func @cast_with_int_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 3 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 10 : i64
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 9 : i32
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 10 : i32
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 111 : i32
// CHECK:           %[[TYPE_OFFSET_0:.*]] = ptr.type_offset i32 : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_7]], %[[TYPE_OFFSET_0]] : i32
// CHECK:           %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[UNREALIZED_CONVERSION_CAST_1]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_1:.*]] = ptr.type_offset i8 : i32
// CHECK:           %[[MULI_1:.*]] = arith.muli %[[CONSTANT_6]], %[[TYPE_OFFSET_1]] : i32
// CHECK:           %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[PTR_ADD_0]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[UNREALIZED_CONVERSION_CAST_0]] : !ptr.ptr<#ptr.generic_space> to i64
// CHECK:           %[[TYPE_OFFSET_2:.*]] = ptr.type_offset i32 : i64
// CHECK:           %[[MULI_2:.*]] = arith.muli %[[UNREALIZED_CONVERSION_CAST_2]], %[[TYPE_OFFSET_2]] : i64
// CHECK:           %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[UNREALIZED_CONVERSION_CAST_0]], %[[MULI_2]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           %[[TYPE_OFFSET_3:.*]] = ptr.type_offset i32 : i32
// CHECK:           %[[MULI_3:.*]] = arith.muli %[[CONSTANT_5]], %[[TYPE_OFFSET_3]] : i32
// CHECK:           %[[PTR_ADD_3:.*]] = ptr.ptr_add %[[PTR_ADD_2]], %[[MULI_3]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[PTR_ADD_3]] : !ptr.ptr<#ptr.generic_space> to i64
// CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[UNREALIZED_CONVERSION_CAST_3]], %[[CONSTANT_4]] : i64
// CHECK:           %[[TYPE_OFFSET_4:.*]] = ptr.type_offset i32 : i32
// CHECK:           %[[MULI_4:.*]] = arith.muli %[[CONSTANT_3]], %[[TYPE_OFFSET_4]] : i32
// CHECK:           %[[PTR_ADD_4:.*]] = ptr.ptr_add %[[PTR_ADD_1]], %[[MULI_4]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_5:.*]] = ptr.type_offset i32 : i64
// CHECK:           %[[MULI_5:.*]] = arith.muli %[[REMSI_0]], %[[TYPE_OFFSET_5]] : i64
// CHECK:           %[[PTR_ADD_5:.*]] = ptr.ptr_add %[[PTR_ADD_4]], %[[MULI_5]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           %[[TYPE_OFFSET_6:.*]] = ptr.type_offset i64 : i32
// CHECK:           %[[MULI_6:.*]] = arith.muli %[[CONSTANT_2]], %[[TYPE_OFFSET_6]] : i32
// CHECK:           %[[PTR_ADD_6:.*]] = ptr.ptr_add %[[PTR_ADD_5]], %[[MULI_6]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_7:.*]] = ptr.type_offset i64 : i32
// CHECK:           %[[MULI_7:.*]] = arith.muli %[[ARG2]], %[[TYPE_OFFSET_7]] : i32
// CHECK:           %[[PTR_ADD_7:.*]] = ptr.ptr_add %[[PTR_ADD_6]], %[[MULI_7]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_8:.*]] = ptr.type_offset i64 : i32
// CHECK:           %[[MULI_8:.*]] = arith.muli %[[CONSTANT_1]], %[[TYPE_OFFSET_8]] : i32
// CHECK:           %[[PTR_ADD_8:.*]] = ptr.ptr_add %[[PTR_ADD_7]], %[[MULI_8]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_9:.*]] = ptr.type_offset i16 : i32
// CHECK:           %[[MULI_9:.*]] = arith.muli %[[CONSTANT_0]], %[[TYPE_OFFSET_9]] : i32
// CHECK:           %[[PTR_ADD_9:.*]] = ptr.ptr_add %[[PTR_ADD_8]], %[[MULI_9]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_10:.*]] = ptr.type_offset i16 : i32
// CHECK:           %[[MULI_10:.*]] = arith.muli %[[ARG2]], %[[TYPE_OFFSET_10]] : i32
// CHECK:           %[[PTR_ADD_10:.*]] = ptr.ptr_add %[[PTR_ADD_9]], %[[MULI_10]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[TYPE_OFFSET_11:.*]] = ptr.type_offset i16 : i32
// CHECK:           %[[MULI_11:.*]] = arith.muli %[[CONSTANT_1]], %[[TYPE_OFFSET_11]] : i32
// CHECK:           %[[PTR_ADD_11:.*]] = ptr.ptr_add %[[PTR_ADD_10]], %[[MULI_11]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           %[[LOAD_0:.*]] = ptr.load %[[PTR_ADD_11]] : !ptr.ptr<#ptr.generic_space> -> i32
// CHECK:           %[[EXTSI_0:.*]] = arith.extsi %[[ARG2]] : i32 to i64
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[REMSI_0]], %[[EXTSI_0]] : i64
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_4:.*]] = builtin.unrealized_conversion_cast %[[ADDI_0]] : i64 to !ptr.ptr<#ptr.generic_space>
// CHECK:           ptr.store %[[LOAD_0]], %[[UNREALIZED_CONVERSION_CAST_4]] : i32, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
  tt.func public @cast_with_int_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i64 = arith.constant 10 : i64
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c111_i32 = arith.constant 111 : i32
    %0 = tt.addptr %arg0, %c111_i32 : !tt.ptr<i32>, i32
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<i8>
    %2 = tt.addptr %1, %c10_i32 : !tt.ptr<i8>, i32
    %3 = tt.bitcast %2 : !tt.ptr<i8> -> !tt.ptr<i32>
    %4 = tt.ptr_to_int %arg1 : !tt.ptr<i32> -> i64
    %5 = tt.addptr %arg1, %4 : !tt.ptr<i32>, i64
    %6 = tt.addptr %5, %c9_i32 : !tt.ptr<i32>, i32
    %7 = tt.ptr_to_int %6 : !tt.ptr<i32> -> i64
    %8 = arith.remsi %7, %c10_i64 : i64
    %9 = tt.addptr %3, %c1_i32 : !tt.ptr<i32>, i32
    %10 = tt.addptr %9, %8 : !tt.ptr<i32>, i64
    %11 = tt.bitcast %10 : !tt.ptr<i32> -> !tt.ptr<i64>
    %12 = tt.addptr %11, %c2_i32 : !tt.ptr<i64>, i32
    %13 = tt.addptr %12, %arg2 : !tt.ptr<i64>, i32
    %14 = tt.addptr %13, %c3_i32 : !tt.ptr<i64>, i32
    %15 = tt.bitcast %14 : !tt.ptr<i64> -> !tt.ptr<i16>
    %16 = tt.addptr %15, %c4_i32 : !tt.ptr<i16>, i32
    %17 = tt.addptr %16, %arg2 : !tt.ptr<i16>, i32
    %18 = tt.addptr %17, %c3_i32 : !tt.ptr<i16>, i32
    %19 = tt.bitcast %18 : !tt.ptr<i16> -> !tt.ptr<i32>
    %20 = tt.load %19 : !tt.ptr<i32>
    %21 = arith.extsi %arg2 : i32 to i64
    %22 = arith.addi %8, %21 : i64
    %23 = tt.int_to_ptr %22 : i64 -> !tt.ptr<i32>
    tt.store %23, %20 : !tt.ptr<i32>
    tt.return
  }
}
