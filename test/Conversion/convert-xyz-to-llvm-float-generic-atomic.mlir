// RUN: triton-xyz-opt --convert-xyz-to-llvm %s | FileCheck %s

module {
// CHECK-LABEL:   llvm.func @float_generic_atomic_cas(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG2:[-0-9A-Za-z$._]+]]: i64,
// CHECK-SAME:      %[[ARG3:[-0-9A-Za-z$._]+]]: i64,
// CHECK-SAME:      %[[ARG4:[-0-9A-Za-z$._]+]]: i64,
// CHECK-SAME:      %[[ARG5:[-0-9A-Za-z$._]+]]: i64,
// CHECK-SAME:      %[[ARG6:[-0-9A-Za-z$._]+]]: f32,
// CHECK-SAME:      %[[ARG7:[-0-9A-Za-z$._]+]]: f32) -> f32 {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG0]], %[[MLIR_0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG1]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[ARG3]], %[[INSERTVALUE_2]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[ARG4]], %[[INSERTVALUE_3]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[EXTRACTVALUE_0]]{{\[}}%[[ARG5]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[LOAD_0:.*]] = llvm.load %[[GETELEMENTPTR_0]] : !llvm.ptr -> i32
// CHECK:           llvm.br ^bb1(%[[LOAD_0]] : i32)
// CHECK:         ^bb1(%[[VAL_0:.*]]: i32):
// CHECK:           %[[BITCAST_0:.*]] = llvm.bitcast %[[VAL_0]] : i32 to f32
// CHECK:           %[[FCMP_0:.*]] = llvm.fcmp "oeq" %[[BITCAST_0]], %[[ARG6]] : f32
// CHECK:           %[[SELECT_0:.*]] = llvm.select %[[FCMP_0]], %[[ARG7]], %[[BITCAST_0]] : i1, f32
// CHECK:           %[[BITCAST_1:.*]] = llvm.bitcast %[[SELECT_0]] : f32 to i32
// CHECK:           %[[CMPXCHG_0:.*]] = llvm.cmpxchg %[[GETELEMENTPTR_0]], %[[VAL_0]], %[[BITCAST_1]] acq_rel monotonic : !llvm.ptr, i32
// CHECK:           %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[CMPXCHG_0]][0] : !llvm.struct<(i32, i1)>
// CHECK:           %[[EXTRACTVALUE_2:.*]] = llvm.extractvalue %[[CMPXCHG_0]][1] : !llvm.struct<(i32, i1)>
// CHECK:           llvm.cond_br %[[EXTRACTVALUE_2]], ^bb2, ^bb1(%[[EXTRACTVALUE_1]] : i32)
// CHECK:         ^bb2:
// CHECK:           %[[BITCAST_2:.*]] = llvm.bitcast %[[EXTRACTVALUE_1]] : i32 to f32
// CHECK:           llvm.return %[[BITCAST_2]] : f32
// CHECK:         }
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
