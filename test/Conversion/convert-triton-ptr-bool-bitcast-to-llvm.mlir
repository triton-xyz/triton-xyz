// RUN: triton-xyz-opt --triton-to-ptr --convert-xyz-to-llvm --reconcile-unrealized-casts %s | FileCheck %s

module {
// CHECK-LABEL: llvm.func @bool_ptr_bitcast(
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: !llvm.ptr)
// CHECK-NOT: tt.bitcast
// CHECK-NOT: builtin.unrealized_conversion_cast
  func.func @bool_ptr_bitcast(%arg0: memref<*xi1>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : memref<*xi1> to !tt.ptr<i1>
    %1 = tt.bitcast %0 : !tt.ptr<i1> -> !tt.ptr<i8>
    %2 = builtin.unrealized_conversion_cast %1 : !tt.ptr<i8> to memref<*xi8>
    return
  }
}
