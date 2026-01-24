// RUN: triton-shared-opt --reconcile-ptr-casts --cse --canonicalize %s | FileCheck %s

module {
  func.func @reconcile_casts(%memref: memref<*xf32>, %ptr: !ptr.ptr<#ptr.generic_space>) -> (memref<*xf32>, !ptr.ptr<#ptr.generic_space>) {
    // Check conversion from unranked memref to ptr
    %0 = builtin.unrealized_conversion_cast %memref : memref<*xf32> to !ptr.ptr<#ptr.generic_space>

    // Check conversion from ptr to unranked memref
    %1 = builtin.unrealized_conversion_cast %ptr : !ptr.ptr<#ptr.generic_space> to memref<*xf32>

    return %1, %0 : memref<*xf32>, !ptr.ptr<#ptr.generic_space>
  }
}

// CHECK-LABEL: func.func @reconcile_casts
// CHECK-SAME: ([[ARG0:%.+]]: memref<*xf32>, [[ARG1:%.+]]: !ptr.ptr<#ptr.generic_space>)
// CHECK-DAG:   [[RANKED:%.+]] = memref.cast [[ARG0]] : memref<*xf32> to memref<1xf32>
// CHECK-DAG:   [[MEMSPACE_CAST:%.+]] = memref.memory_space_cast [[RANKED]] : memref<1xf32> to memref<1xf32, #ptr.generic_space>
// CHECK-DAG:   [[TO_PTR:%.+]] = ptr.to_ptr [[MEMSPACE_CAST]] : memref<1xf32, #ptr.generic_space> -> {{.*}}<#ptr.generic_space>
// CHECK-DAG:   [[FROM_PTR:%.+]] = ptr.from_ptr [[ARG1]] : {{.*}}<#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK-DAG:   [[MEMSPACE_CAST_BACK:%.+]] = memref.memory_space_cast [[FROM_PTR]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK-DAG:   [[CAST_BACK:%.+]] = memref.cast [[MEMSPACE_CAST_BACK]] : memref<1xf32> to memref<*xf32>
// CHECK:       return [[CAST_BACK]], [[TO_PTR]]
