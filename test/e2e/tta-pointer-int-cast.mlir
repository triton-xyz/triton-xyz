// RUN: triton-xyz-opt --split-input-file --triton-to-linalg-tta --one-shot-bufferize --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --memref-expand --expand-strided-metadata --convert-xyz-to-llvm --reconcile-unrealized-casts --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   llvm.func @runtime_pointer_store(
// CHECK:           %[[RAW_ADDR:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i64
// CHECK-NOT:       ptr.from_ptr
// CHECK-NOT:       builtin.unrealized_conversion_cast
// CHECK:           %[[RUNTIME_PTR:.*]] = llvm.inttoptr %[[RAW_ADDR]] : i64 to !llvm.ptr
// CHECK:           %[[DESC0:.*]] = llvm.insertvalue %[[RUNTIME_PTR]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[DESC1:.*]] = llvm.insertvalue %[[RUNTIME_PTR]], %[[DESC0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.return
  tt.func @runtime_pointer_store(%ans: !tt.ptr<i64>, %src: !tt.ptr<f32>) {
    %val = tt.load %src : !tt.ptr<f32>
    %raw = tt.load %ans : !tt.ptr<i64>
    %dst = tt.int_to_ptr %raw : i64 -> !tt.ptr<f32>
    tt.store %dst, %val : !tt.ptr<f32>
    tt.return
  }
}
