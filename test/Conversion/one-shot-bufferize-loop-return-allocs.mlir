// RUN: triton-xyz-opt %s --one-shot-bufferize=allow-return-allocs-from-loops | FileCheck %s

module {
// CHECK-LABEL:   func.func @loop_return_allocs(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: index) -> tensor<4xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
// CHECK:           linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[ALLOC_0]] : memref<4xi32>)
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG0]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[ALLOC_0]]) -> (memref<4xi32>) {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : index to i32
// CHECK:             %[[ALLOC_1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
// CHECK:             linalg.fill ins(%[[INDEX_CAST_0]] : i32) outs(%[[ALLOC_1]] : memref<4xi32>)
// CHECK:             scf.yield %[[ALLOC_1]] : memref<4xi32>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[FOR_0]] : memref<4xi32> to tensor<4xi32>
// CHECK:           return %[[TO_TENSOR_0]] : tensor<4xi32>
// CHECK:         }
  func.func @loop_return_allocs(%n: index) -> tensor<4xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<4xi32>
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
    %res = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<4xi32> {
      %iv_i32 = arith.index_cast %iv : index to i32
      %next_empty = tensor.empty() : tensor<4xi32>
      %next = linalg.fill ins(%iv_i32 : i32) outs(%next_empty : tensor<4xi32>) -> tensor<4xi32>
      scf.yield %next : tensor<4xi32>
    }
    return %res : tensor<4xi32>
  }
}

