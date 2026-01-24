// RUN: triton-shared-opt --triton-tensor-ptr-to-linalg --triton-to-ptr %s | FileCheck %s

module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   tt.func @kernel(
// CHECK-SAME:                    %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:                    %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>) {
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
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
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]] : tensor<4x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_4:.*]]: i32):
// CHECK:             %[[LOAD_0:.*]] = ptr.load %[[VAL_3]] : !ptr.ptr<#ptr.generic_space> -> i32
// CHECK:             linalg.yield %[[LOAD_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[SPLAT_1:.*]] = tt.splat %[[ARG1]] : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[SPLAT_1]] : tensor<4x!tt.ptr<i32>> to tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[UNREALIZED_CONVERSION_CAST_1]], %[[MAKE_RANGE_0]] : tensor<4x!ptr.ptr<#ptr.generic_space>>, tensor<4xi32>) outs(%[[UNREALIZED_CONVERSION_CAST_1]] : tensor<4x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_1:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_6]], %[[TYPE_OFFSET_1]] : i32
// CHECK:             %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[VAL_5]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<4x!ptr.ptr<#ptr.generic_space>>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_2]], %[[GENERIC_1]] : tensor<4x!ptr.ptr<#ptr.generic_space>>, tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_9:.*]]: i32):
// CHECK:             ptr.store %[[VAL_9]], %[[VAL_8]] : i32, !ptr.ptr<#ptr.generic_space>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<i32>) -> () {
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %a_splat = tt.splat %a : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %a_ptrs = tt.addptr %a_splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %vals = tt.load %a_ptrs : tensor<4x!tt.ptr<i32>>
    %b_splat = tt.splat %b : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %b_ptrs = tt.addptr %b_splat, %range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %b_ptrs, %vals : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
