// RUN: triton-xyz-opt --split-input-file --convert-xyz-to-llvm %s | FileCheck %s

module {
// CHECK-LABEL:   llvm.func @load_from_generic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) -> i32 {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !llvm.ptr to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[LOAD_0:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> i32
// CHECK:           llvm.return %[[LOAD_0]] : i32
// CHECK:         }
  llvm.func @load_from_generic(%arg0: !llvm.ptr) -> i32 {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !ptr.ptr<#ptr.generic_space>
    %1 = ptr.load %0 : !ptr.ptr<#ptr.generic_space> -> i32
    llvm.return %1 : i32
  }
}

// -----

module {
// CHECK-LABEL:   llvm.func @store_to_generic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !llvm.ptr to !ptr.ptr<#ptr.generic_space>
// CHECK:           llvm.store %[[ARG1]], %[[ARG0]] : i32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }
  llvm.func @store_to_generic(%arg0: !llvm.ptr, %arg1: i32) {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !ptr.ptr<#ptr.generic_space>
    ptr.store %arg1, %0 : i32, !ptr.ptr<#ptr.generic_space>
    llvm.return
  }
}

// -----

module {
// CHECK-LABEL:   llvm.func @memspace_cast_roundtrip(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> i32 {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG0]], %[[MLIR_0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG1]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[ARG3]], %[[INSERTVALUE_2]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[ARG4]], %[[INSERTVALUE_3]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[MLIR_1:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_2:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_3:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_4:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[ADDRSPACECAST_0:.*]] = llvm.addrspacecast %[[EXTRACTVALUE_0]] : !llvm.ptr to !llvm.ptr
// CHECK:           %[[ADDRSPACECAST_1:.*]] = llvm.addrspacecast %[[EXTRACTVALUE_1]] : !llvm.ptr to !llvm.ptr
// CHECK:           %[[MLIR_2:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[ADDRSPACECAST_0]], %[[MLIR_2]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_6:.*]] = llvm.insertvalue %[[ADDRSPACECAST_1]], %[[INSERTVALUE_5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_7:.*]] = llvm.insertvalue %[[EXTRACTVALUE_2]], %[[INSERTVALUE_6]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_8:.*]] = llvm.insertvalue %[[EXTRACTVALUE_3]], %[[INSERTVALUE_7]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_9:.*]] = llvm.insertvalue %[[EXTRACTVALUE_4]], %[[INSERTVALUE_8]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_5:.*]] = llvm.extractvalue %[[INSERTVALUE_9]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[LOAD_0:.*]] = llvm.load %[[EXTRACTVALUE_5]] : !llvm.ptr -> i32
// CHECK:           llvm.return %[[LOAD_0]] : i32
// CHECK:         }
  func.func @memspace_cast_roundtrip(%arg0: memref<1xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %0 = memref.memory_space_cast %arg0 : memref<1xi32> to memref<1xi32, #ptr.generic_space>
    %1 = ptr.to_ptr %0 : memref<1xi32, #ptr.generic_space> -> <#ptr.generic_space>
    %2 = ptr.load %1 : !ptr.ptr<#ptr.generic_space> -> i32
    return %2 : i32
  }
}
