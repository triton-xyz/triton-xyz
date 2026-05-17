// RUN: triton-xyz-opt --split-input-file --tta-to-memref %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed_non_zero_direct_step(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<4xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG3]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>)  : i32 {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : tensor<4xi32> to tensor<4xindex>
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi eq, %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[SPLAT_0:.*]] = tensor.splat %[[INDEX_CAST_0]] : tensor<4xindex>
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_1]], %[[SPLAT_0]] : tensor<4xindex>
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:             %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<4xindex>) {
// CHECK:               scf.yield %[[CONSTANT_3]] : tensor<4xindex>
// CHECK:             } else {
// CHECK:               scf.yield %[[MULI_0]] : tensor<4xindex>
// CHECK:             }
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 4 : index
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[IF_0]]{{\[}}%[[VAL_3]]] : tensor<4xindex>
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[EXTRACT_0]], %[[INDEX_CAST_0]] : index
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[ADDI_0]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]]{{\[}}%[[VAL_3]]] [1] [1] : memref<4xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               memref.copy %[[REINTERPRET_CAST_0]], %[[SUBVIEW_0]] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             }
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[INDEX_CAST_3:.*]] = arith.index_cast %[[ARG2]] : tensor<4xi32> to tensor<4xindex>
// CHECK:             %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[INDEX_CAST_2]], %[[CONSTANT_7]] : index
// CHECK:             %[[SPLAT_1:.*]] = tensor.splat %[[INDEX_CAST_2]] : tensor<4xindex>
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[INDEX_CAST_3]], %[[SPLAT_1]] : tensor<4xindex>
// CHECK:             %[[CONSTANT_8:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:             %[[IF_1:.*]] = scf.if %[[CMPI_1]] -> (tensor<4xindex>) {
// CHECK:               scf.yield %[[CONSTANT_8]] : tensor<4xindex>
// CHECK:             } else {
// CHECK:               scf.yield %[[MULI_1]] : tensor<4xindex>
// CHECK:             }
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[CONSTANT_9:.*]] = arith.constant 4 : index
// CHECK:             %[[CONSTANT_10:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_10]] to %[[CONSTANT_9]] step %[[CONSTANT_11]] {
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[IF_1]]{{\[}}%[[VAL_4]]] : tensor<4xindex>
// CHECK:               %[[ADDI_1:.*]] = arith.addi %[[EXTRACT_1]], %[[INDEX_CAST_2]] : index
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[ADDI_1]]], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[TO_TENSOR_0]]{{\[}}%[[VAL_4]]] [1] [1] : tensor<4xf32> to tensor<1xf32>
// CHECK:               bufferization.materialize_in_destination %[[EXTRACT_SLICE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<1xf32>, memref<1xf32, strided<[1], offset: ?>>) -> ()
// CHECK:             }
// CHECK:             %[[VAL_5:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_6:.*]] = "tta.indirect_reindex"(%[[VAL_5]], %[[ARG2]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_7:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
// CHECK:             %[[VAL_8:.*]] = "tta.indirect_reindex"(%[[VAL_7]], %[[ARG2]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
// CHECK:             scf.yield %[[VAL_6]], %[[VAL_8]] : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_no_seed_non_zero_direct_step(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %idx: tensor<4xi32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %addr0 = tta.make_addr %src to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %out0 = tta.make_addr %dst to sizes: [4], strides: [1], offsets: [0], layout: [0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 1, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %addr0, %out = %out0) -> (!tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>) : i32 {
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>) -> tensor<4xf32>
      "tta.store"(%out, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 1, 1>, tensor<4xf32>) -> ()
      %next_base = "tta.advance"(%addr) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next = "tta.indirect_reindex"(%next_base, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      %next_out_base = "tta.advance"(%out) <{static_deltas = array<i64: 1>}> : (!tta.addr<f32, 1, 1>) -> !tta.addr<f32, 1, 1>
      %next_out = "tta.indirect_reindex"(%next_out_base, %idx) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 1, 1>, tensor<4xi32>) -> !tta.addr<f32, 1, 1>
      scf.yield %next, %next_out : !tta.addr<f32, 1, 1>, !tta.addr<f32, 1, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_mixed_seed_non_zero_direct_step(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant dense<[0, 1]> : tensor<2xi32>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.indirect_reindex"(%[[MAKE_ADDR_0]], %[[CONSTANT_2]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[ARG2]] step %[[CONSTANT_1]] iter_args(%[[VAL_2:.*]] = %[[VAL_0]], %[[VAL_3:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>)  : i32 {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[CONSTANT_2]] : tensor<2xi32> to tensor<2xindex>
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[CONSTANT_3]] : tensor<4xi32> to tensor<4xindex>
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi eq, %[[INDEX_CAST_0]], %[[CONSTANT_4]] : index
// CHECK:             %[[SPLAT_0:.*]] = tensor.splat %[[INDEX_CAST_0]] : tensor<2xindex>
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_1]], %[[SPLAT_0]] : tensor<2xindex>
// CHECK:             %[[INDEX_CAST_3:.*]] = arith.index_cast %[[CONSTANT_2]] : tensor<2xi32> to tensor<2xindex>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_3]], %[[MULI_0]] : tensor<2xindex>
// CHECK:             %[[SPLAT_1:.*]] = tensor.splat %[[INDEX_CAST_0]] : tensor<4xindex>
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[INDEX_CAST_2]], %[[SPLAT_1]] : tensor<4xindex>
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
// CHECK:             %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<4xindex>) {
// CHECK:               scf.yield %[[CONSTANT_5]] : tensor<4xindex>
// CHECK:             } else {
// CHECK:               scf.yield %[[MULI_1]] : tensor<4xindex>
// CHECK:             }
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[CAST_0:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_0]] : memref<*xf32> to memref<?xf32>
// CHECK:             %[[CONSTANT_6:.*]] = arith.constant 2 : index
// CHECK:             %[[CONSTANT_7:.*]] = arith.constant 4 : index
// CHECK:             %[[CONSTANT_8:.*]] = arith.constant 2 : index
// CHECK:             %[[MINSI_0:.*]] = arith.minsi %[[CONSTANT_6]], %[[CONSTANT_8]] : index
// CHECK:             %[[CONSTANT_9:.*]] = arith.constant 4 : index
// CHECK:             %[[MINSI_1:.*]] = arith.minsi %[[CONSTANT_7]], %[[CONSTANT_9]] : index
// CHECK:             %[[CONSTANT_10:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_10]] to %[[MINSI_0]] step %[[CONSTANT_11]] {
// CHECK:               scf.for %[[VAL_5:.*]] = %[[CONSTANT_10]] to %[[MINSI_1]] step %[[CONSTANT_11]] {
// CHECK:                 %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK:                 %[[EXTRACT_0:.*]] = tensor.extract %[[ADDI_0]]{{\[}}%[[VAL_4]]] : tensor<2xindex>
// CHECK:                 %[[CONSTANT_13:.*]] = arith.constant 4 : index
// CHECK:                 %[[CONSTANT_14:.*]] = arith.constant 0 : index
// CHECK:                 %[[MULI_2:.*]] = arith.muli %[[EXTRACT_0]], %[[CONSTANT_13]] : index
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[CONSTANT_14]], %[[MULI_2]] : index
// CHECK:                 %[[ADDI_2:.*]] = arith.addi %[[CONSTANT_12]], %[[ADDI_1]] : index
// CHECK:                 %[[EXTRACT_1:.*]] = tensor.extract %[[IF_0]]{{\[}}%[[VAL_5]]] : tensor<4xindex>
// CHECK:                 %[[CONSTANT_15:.*]] = arith.constant 1 : index
// CHECK:                 %[[MULI_3:.*]] = arith.muli %[[EXTRACT_1]], %[[CONSTANT_15]] : index
// CHECK:                 %[[ADDI_3:.*]] = arith.addi %[[INDEX_CAST_0]], %[[MULI_3]] : index
// CHECK:                 %[[ADDI_4:.*]] = arith.addi %[[ADDI_2]], %[[ADDI_3]] : index
// CHECK:                 %[[LOAD_0:.*]] = memref.load %[[CAST_0]]{{\[}}%[[ADDI_4]]] : memref<?xf32>
// CHECK:                 memref.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_4]], %[[VAL_5]]] : memref<2x4xf32>
// CHECK:               }
// CHECK:             }
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             %[[INDEX_CAST_4:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_4]]], sizes: [2, 4], strides: [4, 1] : memref<*xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_0]] : (tensor<2x4xf32>, memref<2x4xf32, strided<[4, 1], offset: ?>>) -> ()
// CHECK:             %[[VAL_6:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:             %[[VAL_7:.*]] = "tta.indirect_reindex"(%[[VAL_6]], %[[CONSTANT_2]]) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
// CHECK:             %[[VAL_8:.*]] = "tta.indirect_reindex"(%[[VAL_7]], %[[CONSTANT_3]]) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
// CHECK:             %[[VAL_9:.*]] = "tta.advance"(%[[VAL_3]]) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:             scf.yield %[[VAL_8]], %[[VAL_9]] : !tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_mixed_seed_non_zero_direct_step(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx0 = arith.constant dense<[0, 1]> : tensor<2xi32>
    %idx1 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %base = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %seed = "tta.indirect_reindex"(%base, %idx0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
    %out0 = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %seed, %out = %out0) -> (!tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>) : i32 {
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>
      "tta.store"(%out, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
      %next_base = "tta.advance"(%addr) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
      %next0 = "tta.indirect_reindex"(%next_base, %idx0) <{indirect_dim = 0 : i32}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>) -> !tta.addr<f32, 2, 1>
      %next1 = "tta.indirect_reindex"(%next0, %idx1) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<4xi32>) -> !tta.addr<f32, 2, 1>
      %next_out = "tta.advance"(%out) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
      scf.yield %next1, %next_out : !tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>
    }
    tt.return
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_no_seed_dynamic_non_zero_direct_step(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi32>,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<?xi1>,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[MAKE_ADDR_1:.*]] = tta.make_addr %[[ARG1]] to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[ARG4]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[MAKE_ADDR_0]], %[[VAL_2:.*]] = %[[MAKE_ADDR_1]]) -> (!tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>)  : i32 {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG2]] : tensor<?xi32> to tensor<?xindex>
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi eq, %[[INDEX_CAST_0]], %[[CONSTANT_2]] : index
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK:             %[[DIM_0:.*]] = tensor.dim %[[INDEX_CAST_1]], %[[CONSTANT_3]] : tensor<?xindex>
// CHECK:             %[[SPLAT_0:.*]] = tensor.splat %[[INDEX_CAST_0]]{{\[}}%[[DIM_0]]] : tensor<?xindex>
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[INDEX_CAST_1]], %[[SPLAT_0]] : tensor<?xindex>
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:             %[[DIM_1:.*]] = tensor.dim %[[INDEX_CAST_1]], %[[CONSTANT_4]] : tensor<?xindex>
// CHECK:             %[[GENERATE_0:.*]] = tensor.generate %[[DIM_1]] {
// CHECK:             ^bb0(%[[VAL_3:.*]]: index):
// CHECK:               tensor.yield %[[VAL_3]] : index
// CHECK:             } : tensor<?xindex>
// CHECK:             %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<?xindex>) {
// CHECK:               scf.yield %[[VAL_4:.*]] : tensor<?xindex>
// CHECK:             } else {
// CHECK:               scf.yield %[[MULI_0]] : tensor<?xindex>
// CHECK:             }
// CHECK:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK:             %[[DIM_2:.*]] = tensor.dim %[[ARG3]], %[[CONSTANT_5]] : tensor<?xi1>
// CHECK:             %[[GENERATE_1:.*]] = tensor.generate %[[DIM_2]] {
// CHECK:             ^bb0(%[[VAL_5:.*]]: index):
// CHECK:               %[[CONSTANT_6:.*]] = arith.constant true
// CHECK:               tensor.yield %[[CONSTANT_6]] : i1
// CHECK:             } : tensor<?xi1>
// CHECK:             %[[IF_1:.*]] = scf.if %[[CMPI_0]] -> (tensor<?xi1>) {
// CHECK:               scf.yield %[[VAL_6:.*]] : tensor<?xi1>
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG3]] : tensor<?xi1>
// CHECK:             }
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK:             %[[DIM_3:.*]] = tensor.dim %[[IF_0]], %[[CONSTANT_7]] : tensor<?xindex>
// CHECK:             %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTANT_9:.*]] = arith.constant 4 : index
// CHECK:             %[[MINSI_0:.*]] = arith.minsi %[[CONSTANT_9]], %[[DIM_3]] : index
// CHECK:             %[[CONSTANT_10:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_7:.*]] = %[[CONSTANT_8]] to %[[MINSI_0]] step %[[CONSTANT_10]] {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[IF_1]]{{\[}}%[[VAL_7]]] : tensor<?xi1>
// CHECK:               scf.if %[[EXTRACT_0]] {
// CHECK:                 %[[EXTRACT_1:.*]] = tensor.extract %[[IF_0]]{{\[}}%[[VAL_7]]] : tensor<?xindex>
// CHECK:                 %[[ADDI_0:.*]] = arith.addi %[[EXTRACT_1]], %[[INDEX_CAST_0]] : index
// CHECK:                 %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[ADDI_0]]], sizes: [2, 1], strides: [4, 1] : memref<*xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:                 %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_0]][0, %[[VAL_7]]] [2, 1] [1, 1] : memref<2x4xf32> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:                 %[[SUBVIEW_1:.*]] = memref.subview %[[REINTERPRET_CAST_0]][0, 0] [2, 1] [1, 1] : memref<2x1xf32, strided<[4, 1], offset: ?>> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:                 memref.copy %[[SUBVIEW_1]], %[[SUBVIEW_0]] : memref<2x1xf32, strided<[4, 1], offset: ?>> to memref<2x1xf32, strided<[4, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:             %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_2]]], sizes: [2, 4], strides: [4, 1] : memref<*xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<2x4xf32>, memref<2x4xf32, strided<[4, 1], offset: ?>>) -> ()
// CHECK:             %[[VAL_8:.*]] = "tta.advance"(%[[VAL_1]]) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:             %[[VAL_9:.*]] = "tta.indirect_reindex"(%[[VAL_8]], %[[ARG2]], %[[ARG3]]) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 2, 1>
// CHECK:             %[[VAL_10:.*]] = "tta.advance"(%[[VAL_2]]) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:             scf.yield %[[VAL_9]], %[[VAL_10]] : !tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
  tt.func @loop_carried_addr_supported_indirect_recurrence_multi_dim_no_seed_dynamic_non_zero_direct_step(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %idx_dyn: tensor<?xi32>, %mask_dyn: tensor<?xi1>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %base = tta.make_addr %src to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %out0 = tta.make_addr %dst to sizes: [2, 4], strides: [4, 1], offsets: [0, 0], layout: [0, 0] {layout_kind = "strided"} : <f32> to !tta.addr<f32, 2, 1>
    %res:2 = scf.for %iv = %c0 to %n step %c1 iter_args(%addr = %base, %out = %out0) -> (!tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>) : i32 {
      %v = "tta.load"(%addr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>) -> tensor<2x4xf32>
      "tta.store"(%out, %v) <{static_mask_dims = array<i64>}> : (!tta.addr<f32, 2, 1>, tensor<2x4xf32>) -> ()
      %next_base = "tta.advance"(%addr) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
      %next = "tta.indirect_reindex"(%next_base, %idx_dyn, %mask_dyn) <{indirect_dim = 1 : i32}> : (!tta.addr<f32, 2, 1>, tensor<?xi32>, tensor<?xi1>) -> !tta.addr<f32, 2, 1>
      %next_out = "tta.advance"(%out) <{static_deltas = array<i64: 0, 1>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
      scf.yield %next, %next_out : !tta.addr<f32, 2, 1>, !tta.addr<f32, 2, 1>
    }
    tt.return
  }
}
