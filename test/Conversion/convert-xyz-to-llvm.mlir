// RUN: triton-xyz-opt --split-input-file --convert-xyz-to-llvm --reconcile-unrealized-casts --canonicalize --cse %s | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @load_from_generic
  // CHECK-NOT: memref.memory_space_cast
  // CHECK-NOT: ptr.load
  // CHECK-NOT: #ptr.generic_space
  // CHECK: llvm.load
  llvm.func @load_from_generic(%arg0: !llvm.ptr) -> i32 {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !ptr.ptr<#ptr.generic_space>
    %1 = ptr.load %0 : !ptr.ptr<#ptr.generic_space> -> i32
    llvm.return %1 : i32
  }
}

// -----

module {
  // CHECK-LABEL: llvm.func @store_to_generic
  // CHECK-NOT: ptr.store
  // CHECK: llvm.store
  llvm.func @store_to_generic(%arg0: !llvm.ptr, %arg1: i32) {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !ptr.ptr<#ptr.generic_space>
    ptr.store %arg1, %0 : i32, !ptr.ptr<#ptr.generic_space>
    llvm.return
  }
}

// -----

module {
  // CHECK-LABEL: llvm.func @memspace_cast_roundtrip
  // CHECK-NOT: memref.memory_space_cast
  // CHECK-NOT: ptr.load
  // CHECK: llvm.load
  func.func @memspace_cast_roundtrip(%arg0: memref<1xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %0 = memref.memory_space_cast %arg0 : memref<1xi32> to memref<1xi32, #ptr.generic_space>
    %1 = ptr.to_ptr %0 : memref<1xi32, #ptr.generic_space> -> <#ptr.generic_space>
    %2 = ptr.load %1 : !ptr.ptr<#ptr.generic_space> -> i32
    return %2 : i32
  }
}
