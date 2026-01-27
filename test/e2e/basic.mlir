// RUN: triton-shared-opt --split-input-file --triton-to-ptr --reconcile-unrealized-casts --convert-to-llvm --reconcile-unrealized-casts %s | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @ptr_add(
  // CHECK-SAME: !llvm.ptr, i32) -> !llvm.ptr
  // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr) -> !llvm.ptr, i{{[0-9]+}}
  // CHECK: llvm.ptrtoint {{.*}} : !llvm.ptr to i{{[0-9]+}}
  // CHECK: llvm.mul {{.*}} : i{{[0-9]+}}
  // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr, i{{[0-9]+}}) -> !llvm.ptr, i8
  // CHECK: llvm.return {{.*}} : !llvm.ptr
  func.func @ptr_add(%a: !ptr.ptr<#ptr.generic_space>, %idx: i32) -> !ptr.ptr<#ptr.generic_space> {
    %a_tt = builtin.unrealized_conversion_cast %a : !ptr.ptr<#ptr.generic_space> to !tt.ptr<i32>
    %p = tt.addptr %a_tt, %idx : !tt.ptr<i32>, i32
    %p_out = builtin.unrealized_conversion_cast %p : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
    func.return %p_out : !ptr.ptr<#ptr.generic_space>
  }
}
