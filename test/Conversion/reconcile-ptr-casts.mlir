// RUN: triton-xyz-opt --split-input-file --reconcile-ptr-casts %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @reconcile_casts(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !ptr.ptr<#ptr.generic_space>) -> (memref<*xf32>, !ptr.ptr<#ptr.generic_space>) {
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<1xf32>
// CHECK:           %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CAST_0]] : memref<1xf32> to memref<1xf32, #ptr.generic_space>
// CHECK:           %[[TO_PTR_0:.*]] = ptr.to_ptr %[[MEMORY_SPACE_CAST_0]] : memref<1xf32, #ptr.generic_space> -> <#ptr.generic_space>
// CHECK:           %[[FROM_PTR_0:.*]] = ptr.from_ptr %[[ARG1]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK:           %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[FROM_PTR_0]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[MEMORY_SPACE_CAST_1]] : memref<1xf32> to memref<*xf32>
// CHECK:           tt.return %[[CAST_1]], %[[TO_PTR_0]] : memref<*xf32>, !ptr.ptr<#ptr.generic_space>
// CHECK:         }
  tt.func @reconcile_casts(%memref: memref<*xf32>, %ptr: !ptr.ptr<#ptr.generic_space>) -> (memref<*xf32>, !ptr.ptr<#ptr.generic_space>) {
    %0 = builtin.unrealized_conversion_cast %memref : memref<*xf32> to !ptr.ptr<#ptr.generic_space>
    %1 = builtin.unrealized_conversion_cast %ptr : !ptr.ptr<#ptr.generic_space> to memref<*xf32>
    tt.return %1, %0 : memref<*xf32>, !ptr.ptr<#ptr.generic_space>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @reconcile_casts_space(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32, 1>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !ptr.ptr<#ptr.generic_space>) -> (memref<*xf32, 1>, !ptr.ptr<#ptr.generic_space>) {
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xf32, 1> to memref<1xf32, 1>
// CHECK:           %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CAST_0]] : memref<1xf32, 1> to memref<1xf32, #ptr.generic_space>
// CHECK:           %[[TO_PTR_0:.*]] = ptr.to_ptr %[[MEMORY_SPACE_CAST_0]] : memref<1xf32, #ptr.generic_space> -> <#ptr.generic_space>
// CHECK:           %[[FROM_PTR_0:.*]] = ptr.from_ptr %[[ARG1]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK:           %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[FROM_PTR_0]] : memref<1xf32, #ptr.generic_space> to memref<1xf32, 1>
// CHECK:           %[[CAST_1:.*]] = memref.cast %[[MEMORY_SPACE_CAST_1]] : memref<1xf32, 1> to memref<*xf32, 1>
// CHECK:           tt.return %[[CAST_1]], %[[TO_PTR_0]] : memref<*xf32, 1>, !ptr.ptr<#ptr.generic_space>
// CHECK:         }
  tt.func @reconcile_casts_space(%memref: memref<*xf32, 1>, %ptr: !ptr.ptr<#ptr.generic_space>) -> (memref<*xf32, 1>, !ptr.ptr<#ptr.generic_space>) {
    %0 = builtin.unrealized_conversion_cast %memref : memref<*xf32, 1> to !ptr.ptr<#ptr.generic_space>
    %1 = builtin.unrealized_conversion_cast %ptr : !ptr.ptr<#ptr.generic_space> to memref<*xf32, 1>
    tt.return %1, %0 : memref<*xf32, 1>, !ptr.ptr<#ptr.generic_space>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @reconcile_multi_result(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>) -> (!ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]]:2 = builtin.unrealized_conversion_cast %[[ARG0]] : memref<*xf32> to !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>
// CHECK:           tt.return %[[UNREALIZED_CONVERSION_CAST_0]]#0, %[[UNREALIZED_CONVERSION_CAST_0]]#1 : !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>
// CHECK:         }
  tt.func @reconcile_multi_result(%memref: memref<*xf32>) -> (!ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>) {
    %0:2 = builtin.unrealized_conversion_cast %memref : memref<*xf32> to !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>
    tt.return %0#0, %0#1 : !ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>
  }
}
