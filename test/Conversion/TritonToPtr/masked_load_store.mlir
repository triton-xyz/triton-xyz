// RUN: triton-shared-opt --split-input-file --triton-tensor-ptr-to-linalg --triton-to-ptr %s | FileCheck %s

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func public @tensor_ptr(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) attributes {noinline = false} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<2> : tensor<4xi32>
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[MAKE_RANGE_0]], %[[CONSTANT_0]] : tensor<4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tt.splat %[[ARG0]] : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[SPLAT_0]] : tensor<4x!tt.ptr<i32>> to tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[UNREALIZED_CONVERSION_CAST_0]], %[[MAKE_RANGE_0]] : tensor<4x!ptr.ptr<#ptr.generic_space>>, tensor<4xi32>) outs(%[[UNREALIZED_CONVERSION_CAST_0]] : tensor<4x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_0:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[TYPE_OFFSET_0]] : i32
// CHECK:             %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[VAL_0]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[CMPI_0]] : tensor<4x!ptr.ptr<#ptr.generic_space>>, tensor<4xi1>) outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i32):
// CHECK:             %[[IF_0:.*]] = scf.if %[[VAL_4]] -> (i32) {
// CHECK:               %[[LOAD_0:.*]] = ptr.load %[[VAL_3]] : !ptr.ptr<#ptr.generic_space> -> i32
// CHECK:               scf.yield %[[LOAD_0]] : i32
// CHECK:             } else {
// CHECK:               %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:               scf.yield %[[CONSTANT_1]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[IF_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[SPLAT_1]] : tensor<4x!tt.ptr<i32>> to tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[UNREALIZED_CONVERSION_CAST_1]], %[[MAKE_RANGE_0]] : tensor<4x!ptr.ptr<#ptr.generic_space>>, tensor<4xi32>) outs(%[[UNREALIZED_CONVERSION_CAST_1]] : tensor<4x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_1:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_7]], %[[TYPE_OFFSET_1]] : i32
// CHECK:             %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[VAL_6]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_2]], %[[GENERIC_1]], %[[CMPI_0]] : tensor<4x!ptr.ptr<#ptr.generic_space>>, tensor<4xi32>, tensor<4xi1>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i1):
// CHECK:             scf.if %[[VAL_11]] {
// CHECK:               ptr.store %[[VAL_10]], %[[VAL_9]] : i32, !ptr.ptr<#ptr.generic_space>
// CHECK:             }
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %mask = arith.cmpi slt, %range, %cst : tensor<4xi32>
    %src_splat = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %src_splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %vals = tt.load %src_ptrs, %mask : tensor<4x!tt.ptr<i32>>
    %dst_splat = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %dst_ptrs, %vals, %mask : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
