// RUN: triton-xyz-opt --convert-xyz-to-llvm %s | FileCheck %s

module {
// CHECK-LABEL: llvm.func @float_generic_atomic_cas(
// CHECK:       %[[INIT:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:       llvm.br ^bb1(%[[INIT]] : i32)
// CHECK:     ^bb1(%[[BITS:.*]]: i32):
// CHECK:       %[[CUR:.*]] = llvm.bitcast %[[BITS]] : i32 to f32
// CHECK:       %[[NEXT:.*]] = llvm.select %{{.*}}, %{{.*}}, %[[CUR]] : i1, f32
// CHECK:       %[[NEXT_BITS:.*]] = llvm.bitcast %[[NEXT]] : f32 to i32
// CHECK:       %[[PAIR:.*]] = llvm.cmpxchg %{{.*}}, %[[BITS]], %[[NEXT_BITS]] acq_rel monotonic : !llvm.ptr, i32
// CHECK:       llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(i32, i1)>
// CHECK-NOT:   llvm.cmpxchg %{{.*}} : !llvm.ptr, f32
// CHECK:     ^bb2:
// CHECK:       llvm.bitcast %{{.*}} : i32 to f32
  func.func @float_generic_atomic_cas(%arg0: memref<?xf32>, %arg1: index, %arg2: f32, %arg3: f32) -> f32 {
    %old = memref.generic_atomic_rmw %arg0[%arg1] : memref<?xf32> {
    ^bb0(%cur: f32):
      %cmp = arith.cmpf oeq, %cur, %arg2 : f32
      %next = arith.select %cmp, %arg3, %cur : f32
      memref.atomic_yield %next : f32
    }
    return %old : f32
  }
}
