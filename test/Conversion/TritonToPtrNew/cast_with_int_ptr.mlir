// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg" --triton-to-ptr --cse --canonicalize %s | FileCheck %s

module {
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

// CHECK-LABEL:  func.func @cast_with_int_ptr
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_111_:%.+]] = arith.constant 111 : i32
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : i32
// CHECK-DAG:       [[CST_9_:%.+]] = arith.constant 9 : i32
// CHECK-DAG:       [[CST_10_1_:%.+]] = arith.constant 10 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK-DAG:       [[VAR_6_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_0_]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           [[VAR_4_:%.+]] = ptr.type_offset i32 : i32
// CHECK:           [[VAR_7_:%.+]] = arith.muli [[VAR_4_]], [[CST_111_]] : i32
// CHECK-DAG:       [[VAR_8_:%.+]] = ptr.ptr_add [[VAR_6_]], [[VAR_7_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           [[VAR_3_:%.+]] = ptr.type_offset i8 : i32
// CHECK:           [[VAR_9_:%.+]] = arith.muli [[VAR_3_]], [[CST_10_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = ptr.ptr_add [[VAR_8_]], [[VAR_9_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK-DAG:       [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[VAR_5_]] : !ptr.ptr<#ptr.generic_space> to i64
// CHECK:           [[VAR_2_:%.+]] = ptr.type_offset i32 : i64
// CHECK:           [[VAR_12_:%.+]] = arith.muli [[VAR_11_]], [[VAR_2_]] : i64
// CHECK-DAG:       [[VAR_13_:%.+]] = ptr.ptr_add [[VAR_5_]], [[VAR_12_]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.muli [[VAR_4_]], [[CST_9_]] : i32
// CHECK:           [[VAR_15_:%.+]] = ptr.ptr_add [[VAR_13_]], [[VAR_14_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_15_]] : !ptr.ptr<#ptr.generic_space> to i64
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.remsi [[VAR_16_]], [[CST_10_1_]] : i64
// CHECK-DAG:       [[VAR_19_:%.+]] = ptr.ptr_add [[VAR_10_]], [[VAR_4_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.muli [[VAR_17_]], [[VAR_2_]] : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = ptr.ptr_add [[VAR_19_]], [[VAR_20_]] : !ptr.ptr<#ptr.generic_space>, i64
// CHECK:           [[VAR_1_:%.+]] = ptr.type_offset i64 : i32
// CHECK:           [[VAR_22_:%.+]] = arith.muli [[VAR_1_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = ptr.ptr_add [[VAR_21_]], [[VAR_22_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.muli [[PARAM_2_]], [[VAR_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = ptr.ptr_add [[VAR_23_]], [[VAR_24_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.muli [[VAR_1_]], [[CST_3_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = ptr.ptr_add [[VAR_25_]], [[VAR_26_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           [[VAR_0_:%.+]] = ptr.type_offset i16 : i32
// CHECK:           [[VAR_28_:%.+]] = arith.muli [[VAR_0_]], [[CST_4_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_29_:%.+]] = ptr.ptr_add [[VAR_27_]], [[VAR_28_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK-DAG:       [[VAR_30_:%.+]] = arith.muli [[PARAM_2_]], [[VAR_0_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_31_:%.+]] = ptr.ptr_add [[VAR_29_]], [[VAR_30_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.muli [[VAR_0_]], [[CST_3_]] : i32
// CHECK:           [[VAR_33_:%.+]] = ptr.ptr_add [[VAR_31_]], [[VAR_32_]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:           [[VAR_34_:%.+]] = ptr.load [[VAR_33_]] : !ptr.ptr<#ptr.generic_space> -> i32
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.extsi [[PARAM_2_]] : i32 to i64
// CHECK:           [[VAR_37_:%.+]] = arith.addi [[VAR_17_]], [[VAR_36_]] : i64
// CHECK:           [[VAR_38_:%.+]] = builtin.unrealized_conversion_cast [[VAR_37_]] : i64 to !ptr.ptr<#ptr.generic_space>
// CHECK:           ptr.store [[VAR_34_]], [[VAR_38_]] : i32, !ptr.ptr<#ptr.generic_space>
// CHECK:           return
// CHECK:         }
