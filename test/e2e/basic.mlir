// RUN: triton-xyz-opt --split-input-file --triton-to-ptr --triton-tt-ptr-to-ptr --reconcile-unrealized-casts --convert-to-llvm %s | FileCheck %s

module {
// CHECK-LABEL:   llvm.func @ptr_add(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> !llvm.ptr {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[MLIR_0]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           %[[PTRTOINT_0:.*]] = llvm.ptrtoint %[[GETELEMENTPTR_0]] : !llvm.ptr to i32
// CHECK:           %[[MUL_0:.*]] = llvm.mul %[[ARG1]], %[[PTRTOINT_0]] : i32
// CHECK:           %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[ARG0]]{{\[}}%[[MUL_0]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:           llvm.return %[[GETELEMENTPTR_1]] : !llvm.ptr
// CHECK:         }
  func.func @ptr_add(%a: !tt.ptr<i32>, %idx: i32) -> !tt.ptr<i32> {
    %p = tt.addptr %a, %idx : !tt.ptr<i32>, i32
    func.return %p : !tt.ptr<i32>
  }
}
