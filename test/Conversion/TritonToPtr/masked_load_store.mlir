// Test that triton load and store with mask are lowered correctly
// (scf.if guarding the load and store)
// RUN: triton-shared-opt --split-input-file --triton-arith-to-linalg --triton-tensor-ptr-to-linalg --triton-to-ptr %s | FileCheck %s

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @tensor_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 16 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_0]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_2]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[UNREALIZED_CONVERSION_CAST_1]] : !ptr.ptr<#ptr.generic_space>) outs(%[[EMPTY_3]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_2]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>) outs(%[[FILL_2]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_0:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_2]], %[[TYPE_OFFSET_0]] : i32
// CHECK:             %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[VAL_1]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_1]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_4]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_5:.*]]: i32):
// CHECK:             %[[LOAD_0:.*]] = ptr.load %[[VAL_4]] : !ptr.ptr<#ptr.generic_space> -> i32
// CHECK:             linalg.yield %[[LOAD_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<16xi64>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_2]] : tensor<16xi32>) outs(%[[EMPTY_5]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i64):
// CHECK:             %[[EXTSI_0:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
// CHECK:             linalg.yield %[[EXTSI_0]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_3]] : tensor<16xi64>) outs(%[[EMPTY_6]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i64, %[[VAL_9:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_8]] : i64 to !ptr.ptr<#ptr.generic_space>
// CHECK:             linalg.yield %[[UNREALIZED_CONVERSION_CAST_2]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[EMPTY_7:.*]] = tensor.empty() : tensor<16xi1>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_1]] : tensor<16xi32>, tensor<16xi32>) outs(%[[EMPTY_7]] : tensor<16xi1>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i1):
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:             linalg.yield %[[CMPI_0]] : i1
// CHECK:           } -> tensor<16xi1>
// CHECK:           %[[EMPTY_8:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_4]], %[[GENERIC_5]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi1>) outs(%[[EMPTY_8]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_14:.*]]: i1, %[[VAL_15:.*]]: i32):
// CHECK:             %[[IF_0:.*]] = scf.if %[[VAL_14]] -> (i32) {
// CHECK:               %[[LOAD_1:.*]] = ptr.load %[[VAL_13]] : !ptr.ptr<#ptr.generic_space> -> i32
// CHECK:               scf.yield %[[LOAD_1]] : i32
// CHECK:             } else {
// CHECK:               %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:               scf.yield %[[CONSTANT_2]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[IF_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[EMPTY_9:.*]] = tensor.empty() : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[FILL_3:.*]] = linalg.fill ins(%[[UNREALIZED_CONVERSION_CAST_0]] : !ptr.ptr<#ptr.generic_space>) outs(%[[EMPTY_9]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_3]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>) outs(%[[FILL_3]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_16:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_1:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_17]], %[[TYPE_OFFSET_1]] : i32
// CHECK:             %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[VAL_16]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[EMPTY_10:.*]] = tensor.empty() : tensor<16xi64>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_7]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_10]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_20:.*]]: i64):
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_19]] : !ptr.ptr<#ptr.generic_space> to i64
// CHECK:             linalg.yield %[[UNREALIZED_CONVERSION_CAST_3]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[EMPTY_11:.*]] = tensor.empty() : tensor<16xi64>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_6]] : tensor<16xi32>) outs(%[[EMPTY_11]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i32, %[[VAL_22:.*]]: i64):
// CHECK:             %[[EXTSI_1:.*]] = arith.extsi %[[VAL_21]] : i32 to i64
// CHECK:             linalg.yield %[[EXTSI_1]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[GENERIC_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_8]], %[[GENERIC_9]] : tensor<16xi64>, tensor<16xi64>) outs(%[[GENERIC_8]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_23:.*]]: i64, %[[VAL_24:.*]]: i64, %[[VAL_25:.*]]: i64):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_23]], %[[VAL_24]] : i64
// CHECK:             linalg.yield %[[ADDI_0]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[EMPTY_12:.*]] = tensor.empty() : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_7]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_12]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_26:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_27:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             linalg.yield %[[VAL_26]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_12:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_11]], %[[FILL_0]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>) outs(%[[GENERIC_11]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_29:.*]]: i32, %[[VAL_30:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_2:.*]] = ptr.type_offset i64 : i32
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[VAL_29]], %[[TYPE_OFFSET_2]] : i32
// CHECK:             %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[VAL_28]], %[[MULI_2]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_2]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[EMPTY_13:.*]] = tensor.empty() : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_13:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_12]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_13]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_31:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_32:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             linalg.yield %[[VAL_31]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[EMPTY_14:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[GENERIC_14:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_10]] : tensor<16xi64>) outs(%[[EMPTY_14]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_33:.*]]: i64, %[[VAL_34:.*]]: i32):
// CHECK:             %[[TRUNCI_0:.*]] = arith.trunci %[[VAL_33]] : i64 to i32
// CHECK:             linalg.yield %[[TRUNCI_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_13]], %[[GENERIC_14]], %[[GENERIC_5]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>, tensor<16xi1>) {
// CHECK:           ^bb0(%[[VAL_35:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_36:.*]]: i32, %[[VAL_37:.*]]: i1):
// CHECK:             scf.if %[[VAL_37]] {
// CHECK:               ptr.store %[[VAL_36]], %[[VAL_35]] : i32, !ptr.ptr<#ptr.generic_space>
// CHECK:             }
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %cst_0 = arith.constant dense<16> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64>
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>>
    %6 = arith.cmpi slt, %0, %cst_0 : tensor<16xi32>
    %7 = tt.load %5, %6 : tensor<16x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %10 = tt.ptr_to_int %9 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64>
    %11 = arith.extsi %7 : tensor<16xi32> to tensor<16xi64>
    %12 = arith.addi %10, %11 : tensor<16xi64>
    %13 = tt.bitcast %9 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>>
    %14 = tt.addptr %13, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %15 = tt.bitcast %14 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>>
    %16 = arith.trunci %12 : tensor<16xi64> to tensor<16xi32>
    tt.store %15, %16, %6 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}
