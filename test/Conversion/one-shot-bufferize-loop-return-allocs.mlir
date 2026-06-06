// RUN: triton-xyz-opt %s --one-shot-bufferize=allow-return-allocs-from-loops | FileCheck %s

module {
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

// CHECK-LABEL: func.func @loop_return_allocs(
// CHECK-SAME:    %[[N:.*]]: index)
// CHECK:       %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<4xi32>
// CHECK:       %[[FOR:.*]] = scf.for {{.*}} to %[[N]] {{.*}} iter_args(%[[ARG:.*]] = %[[ALLOC]]) -> (memref<4xi32>) {
// CHECK:         %[[NEXT:.*]] = memref.alloc() {{.*}} : memref<4xi32>
// CHECK:         scf.yield %[[NEXT]] : memref<4xi32>
// CHECK:       }
// CHECK:       %[[RET:.*]] = bufferization.to_tensor %[[FOR]] : memref<4xi32> to tensor<4xi32>
// CHECK:       return %[[RET]] : tensor<4xi32>
