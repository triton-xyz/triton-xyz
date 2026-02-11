// RUN: triton-xyz-opt --canonicalize --split-input-file %s | FileCheck %s

module {
// CHECK-LABEL:   tt.func @reindex_zero_fold(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[MAKE_ADDR_0]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @reindex_zero_fold(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r = "tta.reindex"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @advance_of_reindex_compose(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [16, 16], strides: [16, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 8, 11>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[VAL_0]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @advance_of_reindex_compose(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [16, 16], strides: [16, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 3, 4>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %a1 = "tta.advance"(%r0) <{static_deltas = array<i64: 5, 7>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %a1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @reindex_of_advance_compose(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [16, 16], strides: [16, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 7, 9>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[VAL_0]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @reindex_of_advance_compose(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [16, 16], strides: [16, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %a0 = "tta.advance"(%base) <{static_deltas = array<i64: 2, 3>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%a0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 5, 6>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @reindex_chain_compose_dynamic(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: index) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG1]], %[[ARG3]] : index
// CHECK:           %[[ADDI_1:.*]] = arith.addi %[[ARG2]], %[[ARG4]] : index
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[ADDI_0]], %[[ADDI_1]]) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>}> : (!tta.addr<f32, 2, 1>, index, index) -> !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[VAL_0]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @reindex_chain_compose_dynamic(%arg0: !tt.ptr<f32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base, %arg1, %arg2) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>}> : (!tta.addr<f32, 2, 1>, index, index) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%r0, %arg3, %arg4) <{operandSegmentSizes = array<i32: 1, 0, 2, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808>}> : (!tta.addr<f32, 2, 1>, index, index) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @advance_zero_fold(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[MAKE_ADDR_0]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @advance_zero_fold(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %a = "tta.advance"(%base) <{static_deltas = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %a : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @reindex_chain_compose_static(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 3, 5>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[VAL_0]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @reindex_chain_compose_static(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1, 2>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%r0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 2, 3>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @advance_chain_compose_static(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.advance"(%[[MAKE_ADDR_0]]) <{static_deltas = array<i64: 7, 10>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[VAL_0]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @advance_chain_compose_static(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %base = tta.make_addr %arg0 to sizes: [8, 8], strides: [8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %a0 = "tta.advance"(%base) <{static_deltas = array<i64: 4, 6>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    %a1 = "tta.advance"(%a0) <{static_deltas = array<i64: 3, 4>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %a1 : !tta.addr<f32, 2, 1>
  }
}

// -----

module {
// CHECK-LABEL:   tt.func @reindex_chain_not_compose_indirect(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<[0, 1]> : tensor<2xi32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<[true, false]> : tensor<2xi1>
// CHECK:           %[[MAKE_ADDR_0:.*]] = tta.make_addr %[[ARG0]] to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_0:.*]] = "tta.reindex"(%[[MAKE_ADDR_0]], %[[CONSTANT_0]], %[[CONSTANT_1]]) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
// CHECK:           %[[VAL_1:.*]] = "tta.reindex"(%[[VAL_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
// CHECK:           tt.return %[[VAL_1]] : !tta.addr<f32, 2, 1>
// CHECK:         }
  tt.func @reindex_chain_not_compose_indirect(%arg0: !tt.ptr<f32>) -> !tta.addr<f32, 2, 1> {
    %idx = arith.constant dense<[0, 1]> : tensor<2xi32>
    %mask = arith.constant dense<[true, false]> : tensor<2xi1>
    %base = tta.make_addr %arg0 to sizes: [2, 2], strides: [2, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to !tta.addr<f32, 2, 1>
    %r0 = "tta.reindex"(%base, %idx, %mask) <{indirect_dim = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0, 1>, static_offsets = array<i64: 0, 0>}> : (!tta.addr<f32, 2, 1>, tensor<2xi32>, tensor<2xi1>) -> !tta.addr<f32, 2, 1>
    %r1 = "tta.reindex"(%r0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1, 0>}> : (!tta.addr<f32, 2, 1>) -> !tta.addr<f32, 2, 1>
    tt.return %r1 : !tta.addr<f32, 2, 1>
  }
}
