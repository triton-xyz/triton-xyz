// RUN: triton-xyz-opt --split-input-file --tta-to-memref %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_dynamic_lower_bound(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[-0-9A-Za-z$._]+]]: i32,
// CHECK-SAME:      %[[ARG3:[-0-9A-Za-z$._]+]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[ARG2]] to %[[ARG3]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[SUBI_0]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[VAL_0]], %[[ARG2]] : i32
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[SUBI_1]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             %[[VAL_3:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_4:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_3]], %[[VAL_4]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_dynamic_lower_bound(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %lb: i32, %n: i32) {
    %c1 = arith.constant 1 : i32
    %src_addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %dst_addr0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %lb to %n step %c1 iter_args(%src_addr = %src_addr0, %dst_addr = %dst_addr0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %val = "tta.load"(%src_addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%dst_addr, %val) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_src = "tta.advance"(%src_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_dst = "tta.advance"(%dst_addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      scf.yield %next_src, %next_dst : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed_dynamic_lower_bound(
// CHECK-SAME:      %[[ARG0:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[-0-9A-Za-z$._]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[-0-9A-Za-z$._]+]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG3:[-0-9A-Za-z$._]+]]: i32,
// CHECK-SAME:      %[[ARG4:[-0-9A-Za-z$._]+]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[ARG3]] to %[[ARG4]] step %[[CONSTANT_0]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[VAL_0]], %[[ARG3]] : i32
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[SUBI_0]] : i32 to index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : tensor<4xi32> to tensor<4xindex>
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi eq, %[[INDEX_CAST_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[SPLAT_0:.*]] = tensor.splat %[[INDEX_CAST_0]] : tensor<4xindex>
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_1]], %[[SPLAT_0]] : tensor<4xindex>
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:             %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<4xindex>) {
// CHECK:               scf.yield %[[CONSTANT_2]] : tensor<4xindex>
// CHECK:             } else {
// CHECK:               scf.yield %[[MULI_0]] : tensor<4xindex>
// CHECK:             }
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_3]] step %[[CONSTANT_5]] {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[IF_0]]{{\[}}%[[VAL_3]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[EXTRACT_0]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_3]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[SUBI_1:.*]] = arith.subi %[[VAL_0]], %[[ARG3]] : i32
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[SUBI_1]] : i32 to index
// CHECK:             %[[INDEX_CAST_3:.*]] = arith.index_cast %[[ARG2]] : tensor<4xi32> to tensor<4xindex>
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[INDEX_CAST_2]], %[[CONSTANT_6]] : index
// CHECK:             %[[SPLAT_1:.*]] = tensor.splat %[[INDEX_CAST_2]] : tensor<4xindex>
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[INDEX_CAST_3]], %[[SPLAT_1]] : tensor<4xindex>
// CHECK:             %[[CONSTANT_7:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:             %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (tensor<4xindex>) {
// CHECK:               scf.yield %[[CONSTANT_7]] : tensor<4xindex>
// CHECK:             } else {
// CHECK:               scf.yield %[[MULI_1]] : tensor<4xindex>
// CHECK:             }
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[CONSTANT_8:.*]] = arith.constant 4 : index
// CHECK:             %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_10:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_9]] to %[[CONSTANT_8]] step %[[CONSTANT_10]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[IF_1]]{{\[}}%[[VAL_4]]] : tensor<4xindex>
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[EXTRACT_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_4]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:             %[[VAL_5:.*]] = "tta.indirect_reindex"(%[[VAL_1]], %[[ARG2]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_6:.*]] = "tta.indirect_reindex"(%[[VAL_2]], %[[ARG2]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_5]], %[[VAL_6]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed_dynamic_lower_bound(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %idx: tensor<4xi32>, %lb: i32, %n: i32) {
    %c1 = arith.constant 1 : i32
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %out0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], wrap_boundaries: [0], layout: "strided" : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %lb to %n step %c1 iter_args(%addr = %addr0, %out = %out0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%out, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next = "tta.indirect_reindex"(%addr, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      %next_out = "tta.indirect_reindex"(%out, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      scf.yield %next, %next_out : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}
