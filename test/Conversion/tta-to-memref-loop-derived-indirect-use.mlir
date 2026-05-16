// RUN: triton-xyz-opt --split-input-file --tta-normalize --tta-to-memref --canonicalize --cse %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_derived_indirect_use(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_4]] to %[[ARG2]] step %[[CONSTANT_5]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_3]]] : tensor<4xindex>
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[EXTRACT_0]], %[[INDEX_CAST_0]] : index
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[ADDI_0]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_3]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[CONSTANT_0]]{{\[}}%[[VAL_4]]] : tensor<4xindex>
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[EXTRACT_1]], %[[INDEX_CAST_0]] : index
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[ADDI_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_4]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:             %[[VAL_5:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_6:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_5]], %[[VAL_6]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_derived_indirect_use(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %out0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %addr0, %out = %out0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %use = "tta.indirect_reindex"(%addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      %out_use = "tta.indirect_reindex"(%out, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      %v = "tta.load"(%use) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%out_use, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_out = "tta.advance"(%out) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next, %next_out : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}
